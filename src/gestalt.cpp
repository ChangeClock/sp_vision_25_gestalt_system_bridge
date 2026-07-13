// gestalt: ORIGINAL sp_vision auto_aim chain (detector/solver/tracker/aimer/
// shooter, byte-identical) driving a UE-simulator sentry turret through the
// GameIO backend (same-frame local shared-memory capture + WebSocket console). Mirrors
// src/standard.cpp's loop; adds the prep-stage setup sequence (spawn pid0
// sentry, drive to the standoff marker, external-turret claim) and the
// auto_aim_test-style "reprojection" debug window.
//
// Usage: gestalt.exe <ws_port> [config] [--setup 1] [--fire 1]
//        [--timeout 800]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/command.hpp"
#include "io/gestalt/game_link.hpp"
#include "io/gestalt/shared_frame_capture.hpp"
#include "io/gestalt/win_capture.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;
namespace ga = io::gestalt::attr;

namespace
{
constexpr double kDeg2Rad = CV_PI / 180.0;
constexpr double kRad2Deg = 180.0 / CV_PI;
// transform_define.csv Outpost2026_0 (cm/100) — coarse bootstrap ONLY, never
// enters the aim path (same convention as tools/outpost_benchmark.py).
constexpr double kRedOutpostX = -3.81, kRedOutpostY = -2.83;
constexpr double kStandoffXcm = 13.0, kStandoffYcm = -306.0;
constexpr int kRedTeamId = 0;
constexpr int kSentryClassId = 1004;
constexpr int kHachisenEntityConfigId = 66000005;

std::chrono::steady_clock::time_point qpc_to_steady(double t_seconds)
{
  // MSVC steady_clock ticks ARE QPC-since-boot scaled to ns, the same clock
  // DXGI LastPresentTime uses — a direct construction stays on now()'s axis.
  return std::chrono::steady_clock::time_point(
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(t_seconds)));
}

bool wait_for(const std::function<bool()> & pred, double timeout_s, double poll_s = 0.5)
{
  auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (std::chrono::steady_clock::now() < t_end) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::duration<double>(poll_s));
  }
  return pred();
}

// A claim is a five-second game-side lease. Keep its renewal independent from
// model inference, shared-memory waits, UI work, and every other potentially
// blocking control-loop segment. GameLink serializes websocket poll/send and
// assigns request ids atomically, so this dedicated sender is safe.
class ExternalAimClaimHeartbeat
{
public:
  explicit ExternalAimClaimHeartbeat(io::gestalt::GameLink & link)
  : link_(link), worker_([this] { run(); })
  {
  }

  ~ExternalAimClaimHeartbeat()
  {
    {
      std::lock_guard<std::mutex> lock(mu_);
      active_ = false;
      stop_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void activate(int player_id)
  {
    std::lock_guard<std::mutex> lock(mu_);
    player_id_ = player_id;
    active_ = true;
    // The initial claim is synchronous: callers may immediately issue
    // RBTakeOver/camera commands knowing mode preservation has been requested.
    link_.exec(fmt::format("ExtAimClaim {} 1", player_id_));
    wake_.notify_all();
  }

  void pause()
  {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = false;
    wake_.notify_all();
  }

private:
  void run()
  {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
      wake_.wait(lock, [&] { return stop_ || active_; });
      if (stop_) break;
      const int scheduled_player = player_id_;
      const bool interrupted = wake_.wait_for(lock, 1s, [&] {
        return stop_ || !active_ || player_id_ != scheduled_player;
      });
      if (interrupted) continue;
      link_.exec(fmt::format("ExtAimClaim {} 1", scheduled_player));
    }
  }

  io::gestalt::GameLink & link_;
  std::mutex mu_;
  std::condition_variable wake_;
  bool active_ = false;
  bool stop_ = false;
  int player_id_ = -1;
  std::thread worker_;
};

// ---- SIM-DOMAIN LIGHTBAR REFINEMENT (bridge/adapter layer) ----
// The yolo11 keypoint head regresses near-symmetric quads on UE renders and
// flattens the perspective asymmetry optimize_yaw discriminates on (raw-pixel
// bar-length ratio modulates +-25-40% with target rotation vs the keypoints'
// pinned 1.01 +- 0.03). Re-extract the two lightbar endpoint pairs from raw
// pixels (color threshold + minAreaRect, the traditional-detector approach)
// inside each detection box; yolo keeps detect/classify. The estimation chain
// (solver/optimize_yaw/tracker/aimer) is untouched.
bool refine_armor_points(const cv::Mat & img, auto_aim::Armor & armor)
{
  std::vector<cv::Point2f> kp(armor.points.begin(), armor.points.end());
  cv::Rect box = cv::boundingRect(kp);
  const int orig_x0 = box.x, orig_x1 = box.x + box.width;
  int pad = std::max(8, box.height / 2);
  box.x -= pad;
  box.y -= pad;
  box.width += 2 * pad;
  box.height += 2 * pad;
  box &= cv::Rect(0, 0, img.cols, img.rows);
  if (box.width < 12 || box.height < 8) return false;

  cv::Mat ch[3];
  cv::split(img(box), ch);
  const bool blue = armor.color == auto_aim::Color::blue;
  const cv::Mat & p = blue ? ch[0] : ch[2];  // primary channel
  const cv::Mat & o = blue ? ch[2] : ch[0];  // opposing channel
  cv::Mat mask = (p > 90) & (p > o * 1.6) & (p > ch[1] * 1.3);

  std::vector<std::vector<cv::Point>> cnts;
  cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  struct Bar
  {
    cv::Point2f top, bot, c;
    float len;
  };
  std::vector<Bar> bars;
  for (const auto & c : cnts) {
    if (cv::contourArea(c) < 4) continue;
    cv::RotatedRect rr = cv::minAreaRect(c);
    const float len = std::max(rr.size.width, rr.size.height);
    const float wid = std::min(rr.size.width, rr.size.height);
    if (len < 5.f || len / std::max(wid, 0.5f) < 1.5f) continue;
    const float ang = rr.angle * static_cast<float>(CV_PI) / 180.f;
    cv::Point2f axis = (rr.size.width >= rr.size.height)
                         ? cv::Point2f(std::cos(ang), std::sin(ang))
                         : cv::Point2f(-std::sin(ang), std::cos(ang));
    if (axis.y < 0) axis = -axis;  // downward
    const cv::Point2f half = axis * (len / 2.f);
    bars.push_back({rr.center - half, rr.center + half, rr.center, len});
  }
  if (bars.size() < 2) return false;
  std::sort(bars.begin(), bars.end(), [](const Bar & a, const Bar & b) { return a.c.x < b.c.x; });
  const Bar & L = bars.front();
  const Bar & R = bars.back();
  if (R.c.x - L.c.x < 6.f) return false;
  const float lr = L.len / std::max(R.len, 1e-3f);
  if (lr < 0.35f || lr > 2.8f) return false;
  // both bars must sit near the yolo box (reject neighbours leaking into the pad)
  const float margin = 0.35f * (orig_x1 - orig_x0);
  if (L.c.x + box.x < orig_x0 - margin || R.c.x + box.x > orig_x1 + margin) return false;

  const cv::Point2f off(static_cast<float>(box.x), static_cast<float>(box.y));
  armor.points[0] = L.top + off;  // TL
  armor.points[1] = R.top + off;  // TR
  armor.points[2] = R.bot + off;  // BR
  armor.points[3] = L.bot + off;  // BL
  armor.center = (armor.points[0] + armor.points[1] + armor.points[2] + armor.points[3]) * 0.25f;
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(
    argc, argv,
    "{@ws-port     |                     | game attribute-WS port}"
    "{@config-path | configs/gestalt.yaml| yaml config}"
    "{pos          | far                 | outpost bench position; far=Blue outpost-lob point (mode 5)}"
    "{mode         | basic               | basic: fixed-axis median filter + known-omega timed fire (zero turret jitter, per-plate z) | ekf: original sp_vision chain}"
    "{record       | 1                   | per-frame est-vs-gt jsonl to gestalt_record.jsonl}"
    "{setup        | 1                   | run prep-stage spawn/drive/claim sequence}"
    "{fire         | 1                   | allow trigger pulls}"
    "{timeout      | 800                 | engagement wall-clock budget, s}"
    "{dumpui       | 0                   | write every Nth annotated debug frame to ui_dump/ (0 = off)}");
  const int port = cli.get<int>("@ws-port");
  const std::string config = cli.get<std::string>("@config-path");
  const std::string pos = cli.get<std::string>("pos");
  const std::string run_mode = cli.get<std::string>("mode");
  const bool mode_basic = run_mode == "basic";
  const bool mode_observe = run_mode == "observe";
  const bool do_record = cli.get<int>("record") != 0;
  (void)mode_basic;  // basic-mode scaffolding pending; observe/ekf active
  const bool do_setup = cli.get<int>("setup") != 0;
  const bool allow_fire = cli.get<int>("fire") != 0;
  const double timeout_s = cli.get<double>("timeout");
  const int dump_ui_every = cli.get<int>("dumpui");
  if (pos != "far") {
    tools::logger()->error(
      "[gestalt] invalid --pos={}: use far (Blue outpost-lob point, AIMoveMode=5)",
      pos);
    return 2;
  }
  if (dump_ui_every > 0) std::filesystem::create_directories("ui_dump");

  tools::logger()->info("[gestalt] port={} config={}", port, config);

  // Config extras (bridge-layer knobs; read BEFORE setup — camera_json is
  // applied during the setup sequence and must match the yaml camera_matrix).
  double sim_quad_scale = 1.192;
  bool z_adapter = false;
  bool bar_refine = false;
  bool v_adapter = false;
  double cfg_delay = 0.08;
  Eigen::Matrix3d R_camera2gimbal_cfg{
    {0, 0, 1},
    {-1, 0, 0},
    {0, -1, 0}};
  std::string camera_json = R"({"enabled":1,"fovDegrees":45,"shutterSpeed":120})";
  int expected_frame_width = 1280;
  int expected_frame_height = 720;
  try {
    auto yroot = YAML::LoadFile(config);
    if (yroot["sim_quad_scale"].IsDefined()) sim_quad_scale = yroot["sim_quad_scale"].as<double>();
    if (yroot["z_adapter"].IsDefined()) z_adapter = yroot["z_adapter"].as<bool>();
    if (yroot["bar_refine"].IsDefined()) bar_refine = yroot["bar_refine"].as<bool>();
    if (yroot["v_adapter"].IsDefined()) v_adapter = yroot["v_adapter"].as<bool>();
    if (yroot["low_speed_delay_time"].IsDefined())
      cfg_delay = yroot["low_speed_delay_time"].as<double>();
    if (yroot["camera_json"].IsDefined()) camera_json = yroot["camera_json"].as<std::string>();
    if (yroot["frame_width"].IsDefined())
      expected_frame_width = yroot["frame_width"].as<int>();
    if (yroot["frame_height"].IsDefined())
      expected_frame_height = yroot["frame_height"].as<int>();
    if (yroot["R_camera2gimbal"].IsDefined()) {
      const auto values = yroot["R_camera2gimbal"].as<std::vector<double>>();
      if (values.size() == 9)
        R_camera2gimbal_cfg =
          Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(values.data());
    }
  } catch (...) {
  }
  double expected_fov_degrees = std::numeric_limits<double>::quiet_NaN();
  double expected_arm_length_cm = std::numeric_limits<double>::quiet_NaN();
  const auto camera_settings = nlohmann::json::parse(camera_json, nullptr, false);
  if (!camera_settings.is_discarded()) {
    if (camera_settings.contains("fovDegrees") && camera_settings["fovDegrees"].is_number())
      expected_fov_degrees = camera_settings["fovDegrees"].get<double>();
    if (camera_settings.contains("armLength") && camera_settings["armLength"].is_number())
      expected_arm_length_cm = camera_settings["armLength"].get<double>();
    else if (
      camera_settings.contains("armLengthCm") && camera_settings["armLengthCm"].is_number())
      expected_arm_length_cm = camera_settings["armLengthCm"].get<double>();
  }

  // ---- bench scenario knobs (external-aim playbook §6). Defaults reproduce
  // the original Mario-vs-outpost far bench, so existing configs are untouched.
  std::string bench_shooter_entity = "66000008";  // Mario assault infantry
  int bench_shooter_class = 1003;                 // sentry bench: 66000012 / 1004
  int bench_allowance = 300;                      // 17mm budget per round
  std::string bench_target = "outpost";           // outpost | vehicle
  int bench_target_pid = 92001;                   // RBNavLab player id of the target
  std::string bench_target_spawn, bench_target_drive, bench_target_spin;
  std::vector<double> bench_target_goal_cm;       // optional arrival ball (x,y cm)
  double bench_target_goal_radius_cm = 70.0;      // hero-class chassis stop ~3m out: use 350
  std::vector<double> bench_target_park_cm;       // optional teardown parking spot (x,y cm)
  int bench_target_hp = 0;                        // >0: boost so the round can't kill it
  double bench_damage_per_hit = 20.0;             // HP per damaging 17mm hit
  double bench_boot_pitch = 10.0;                 // outpost drum sits high; ground cars: 0
  bool bench_match = false;  // live-match takeover: spawn+claim pid 0, then start the match
  int bench_shooter_team = 1;  // 1 = blue (default), 0 = red
  try {
    auto yroot = YAML::LoadFile(config);
    auto ystr = [&](const char * k, std::string & dst) {
      if (yroot[k].IsDefined()) dst = yroot[k].as<std::string>();
    };
    ystr("bench_shooter_entity", bench_shooter_entity);
    if (yroot["bench_shooter_class"].IsDefined())
      bench_shooter_class = yroot["bench_shooter_class"].as<int>();
    if (yroot["bench_allowance"].IsDefined())
      bench_allowance = yroot["bench_allowance"].as<int>();
    ystr("bench_target", bench_target);
    if (yroot["bench_target_pid"].IsDefined())
      bench_target_pid = yroot["bench_target_pid"].as<int>();
    ystr("bench_target_spawn", bench_target_spawn);
    ystr("bench_target_drive", bench_target_drive);
    ystr("bench_target_spin", bench_target_spin);
    if (yroot["bench_target_goal_cm"].IsDefined())
      bench_target_goal_cm = yroot["bench_target_goal_cm"].as<std::vector<double>>();
    if (yroot["bench_target_goal_radius_cm"].IsDefined())
      bench_target_goal_radius_cm = yroot["bench_target_goal_radius_cm"].as<double>();
    if (yroot["bench_target_park_cm"].IsDefined())
      bench_target_park_cm = yroot["bench_target_park_cm"].as<std::vector<double>>();
    if (yroot["bench_target_hp"].IsDefined())
      bench_target_hp = yroot["bench_target_hp"].as<int>();
    if (yroot["bench_damage_per_hit"].IsDefined())
      bench_damage_per_hit = yroot["bench_damage_per_hit"].as<double>();
    if (yroot["bench_boot_pitch"].IsDefined())
      bench_boot_pitch = yroot["bench_boot_pitch"].as<double>();
    if (yroot["bench_match"].IsDefined()) bench_match = yroot["bench_match"].as<bool>();
    if (yroot["bench_shooter_team"].IsDefined())
      bench_shooter_team = yroot["bench_shooter_team"].as<int>();
  } catch (...) {
  }
  int bench_shooter_entity_id = -1;
  try {
    size_t parsed = 0;
    bench_shooter_entity_id = std::stoi(bench_shooter_entity, &parsed);
    if (parsed != bench_shooter_entity.size()) bench_shooter_entity_id = -1;
  } catch (...) {
    bench_shooter_entity_id = -1;
  }
  const bool bench_vehicle = !bench_match && bench_target == "vehicle";
  if (
    bench_match &&
    (camera_settings.is_discarded() || expected_frame_width <= 0 || expected_frame_height <= 0 ||
     !std::isfinite(expected_fov_degrees) || !std::isfinite(expected_arm_length_cm) ||
     std::abs(expected_arm_length_cm) > 0.01)) {
    tools::logger()->error(
      "[gestalt] invalid match camera contract: require valid camera_json with fovDegrees and "
      "armLength=0 plus positive frame_width/frame_height");
    return 2;
  }
  if (
    bench_match &&
    (bench_shooter_entity_id != kHachisenEntityConfigId ||
     bench_shooter_class != kSentryClassId || bench_shooter_team != kRedTeamId)) {
    tools::logger()->error(
      "[gestalt] match config must be red HACHISEN: entity={} class={} team={}",
      bench_shooter_entity, bench_shooter_class, bench_shooter_team);
    return 2;
  }

  io::gestalt::GameLink link(port);
  if (!wait_for([&] { return link.connected(); }, 10)) {
    tools::logger()->error("[gestalt] WS connect failed :{}", port);
    return 1;
  }
  std::this_thread::sleep_for(2s);  // let the initial attribute sweep land

  // Load and compile OpenVINO plus construct the complete solve/track/aim/fire
  // chain before any match-start command. A model load failure therefore
  // leaves the game in prep instead of consuming live match time.
  tools::logger()->info("[gestalt] preparing vision model and solver before takeover");
  auto_aim::YOLO detector(config, false);
  auto_aim::Solver solver(config);
  auto_aim::Tracker tracker(config, solver);
  auto_aim::Aimer aimer(config);
  auto_aim::Shooter shooter(config);
  // Exercise the first OpenVINO request while MatchStatus is still prep; some
  // backends defer allocations/JIT work until infer(), despite compile_model
  // having completed in the constructor.
  const cv::Mat warmup_frame(
    expected_frame_height, expected_frame_width, CV_8UC3, cv::Scalar::all(0));
  const auto warmup_detections = detector.detect(warmup_frame);
  tools::logger()->info(
    "[gestalt] vision model and solver ready (warmup detections={})",
    warmup_detections.size());

  int sentry = -1;   // attribute-map id of the vehicle whose turret we drive
  int ext_pid = 0;   // player id addressed by RBExtAim/ExtAimClaim (bench: pid 0)
  int outpost = -1;
  io::gestalt::SharedFrameCapture cap;
  bool cap_initialized = false;
  uint32_t verified_takeover_epoch = 0;
  bool frame_identity_contract_ok = false;
  ExternalAimClaimHeartbeat claim_heartbeat(link);
  bool identity_contract_ok = true;
  // A natural match settlement can occur while the controlled sentry is dead
  // and its combat map is being recycled. In that state there is no live
  // control target whose mode can be observed/restored; identity was already
  // proven for every live incarnation and the stopped lease is the safety gate.
  bool terminal_inactive_release_allowed = false;

  auto is_expected_match_sentry = [&](int map_id) {
    return map_id > 0 &&
           static_cast<int>(link.attr(map_id, ga::kPlayerId).value_or(-1)) == 0 &&
           static_cast<int>(link.attr(map_id, ga::kTeamId).value_or(-1)) == kRedTeamId &&
           static_cast<int>(link.attr(map_id, ga::kClass).value_or(-1)) == kSentryClassId &&
           link.player_entity_config(0).value_or(-1) == kHachisenEntityConfigId;
  };

  auto ensure_frame_stream = [&]() {
    if (cap_initialized) return true;
    cap_initialized = cap.init(port, 15000);
    if (!cap_initialized) {
      tools::logger()->error(
        "[gestalt] local frame stream unavailable; ensure the rebuilt game is running and "
        "r.VisionBridge.Enable=1");
      return false;
    }
    tools::logger()->info(
      "[gestalt] local frame stream '{}' pid={}", cap.window_title(), cap.process_id());
    // Restore once so a minimized viewport resumes rendering. Do not keep
    // forcing topmost/focus during the control loop.
    cap.raise_window();
    return true;
  };

  auto verify_frame_contract = [&](const char * phase) {
    constexpr int kRequiredConsecutiveFrames = 3;
    frame_identity_contract_ok = false;
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    int consecutive = 0;
    uint64_t previous_frame_id = 0;
    double previous_present = 0;
    uint32_t consecutive_epoch = 0;
    io::gestalt::CapturedFrame observed;
    while (std::chrono::steady_clock::now() < deadline) {
      auto frame = cap.grab(500);
      if (!frame.ok) continue;
      const bool strictly_new =
        (previous_frame_id == 0 || frame.frame_id > previous_frame_id) &&
        (previous_present == 0 || frame.t_present > previous_present);
      previous_frame_id = frame.frame_id;
      previous_present = frame.t_present;
      observed = frame;
      const bool identity_match =
        (frame.identity_flags & 0x7u) == 0x7u && frame.takeover_player_id == ext_pid &&
        frame.takeover_attribute_map_id == sentry && frame.view_actor_unique_id != 0 &&
        frame.view_actor_unique_id == frame.takeover_target_unique_id &&
        frame.takeover_epoch != 0;
      const bool settings_match =
        frame.has_camera_pose && frame.writer_process_id == cap.process_id() &&
        frame.img.cols == expected_frame_width &&
        frame.img.rows == expected_frame_height &&
        std::abs(frame.fov_degrees - expected_fov_degrees) <= 0.10 &&
        std::abs(frame.camera_arm_length_cm - expected_arm_length_cm) <= 0.10 && identity_match;
      if (strictly_new && settings_match) {
        if (consecutive == 0 || consecutive_epoch != frame.takeover_epoch) {
          consecutive = 1;
          consecutive_epoch = frame.takeover_epoch;
        } else {
          ++consecutive;
        }
      } else {
        consecutive = 0;
        consecutive_epoch = 0;
      }
      if (consecutive >= kRequiredConsecutiveFrames) {
        verified_takeover_epoch = consecutive_epoch;
        frame_identity_contract_ok = true;
        tools::logger()->info(
          "[gestalt] {} frame gate passed: {} consecutive frames {}x{} fov={:.2f} "
          "arm={:.2f}cm pid={} map={} actor={} epoch={} flags=0x{:X}",
          phase, consecutive, frame.img.cols, frame.img.rows, frame.fov_degrees,
          frame.camera_arm_length_cm, frame.takeover_player_id,
          frame.takeover_attribute_map_id, frame.view_actor_unique_id,
          frame.takeover_epoch, frame.identity_flags);
        return true;
      }
    }
    tools::logger()->error(
      "[gestalt] {} frame gate failed: got {}x{} fov={:.2f} arm={:.2f}cm frame={} "
      "expected {}x{} fov={:.2f} arm={:.2f}cm; identity pid={} map={} view={} target={} "
      "epoch={} flags=0x{:X}",
      phase, observed.img.cols, observed.img.rows, observed.fov_degrees,
      observed.camera_arm_length_cm, observed.frame_id, expected_frame_width,
      expected_frame_height, expected_fov_degrees, expected_arm_length_cm,
      observed.takeover_player_id, observed.takeover_attribute_map_id,
      observed.view_actor_unique_id, observed.takeover_target_unique_id,
      observed.takeover_epoch, observed.identity_flags);
    return false;
  };

  auto angle_error_degrees = [](double actual, double expected) {
    return std::abs(std::remainder(actual - expected, 360.0));
  };

  auto verify_takeover_contract = [&](const char * phase) {
    if (!wait_for(
          [&] {
            return static_cast<int>(link.attr(sentry, ga::kTargetMode).value_or(-1)) == 90;
          },
          5, 0.1)) {
      tools::logger()->error(
        "[gestalt] {} takeover gate failed: map={} AITargetMode={} (expected 90)", phase,
        sentry, link.attr(sentry, ga::kTargetMode).value_or(-1));
      return false;
    }
    // RBTakeOver uses a 0.3 s view blend. Do not let three residual frames
    // from the previous pawn satisfy a post-revive camera contract.
    std::this_thread::sleep_for(500ms);
    if (!ensure_frame_stream() || !verify_frame_contract(phase)) return false;

    // An in-place revive publishes HP>0 before the formerly hidden vehicle has
    // necessarily recovered from its death pose. RBExtAim's world-to-motor
    // conversion deliberately uses the game's flat-ground approximation, so a
    // nearly vertical muzzle (71 degrees observed in a real HACHISEN revive)
    // is not a valid actuator test even though the command and Hive handle are
    // already available. Wait for the game-owned recovery gates and a sane
    // world-pitch envelope before applying the strict two-axis effect probe.
    constexpr double kTakeoverProbePitchEnvelopeDegrees = 55.0;
    std::optional<double> ready_yaw, ready_pitch;
    const bool actuator_ready = wait_for(
      [&] {
        ready_yaw = link.attr(sentry, ga::kTurretYaw);
        ready_pitch = link.attr(sentry, ga::kTurretPitch);
        return link.attr(sentry, ga::kHealth).value_or(0) > 0 &&
               static_cast<int>(link.attr(sentry, ga::kDefeated).value_or(1)) == 0 &&
               static_cast<int>(link.attr(sentry, ga::kCanOperate).value_or(0)) == 1 &&
               static_cast<int>(link.attr(sentry, ga::kIsChassisOnline).value_or(0)) == 1 &&
               static_cast<int>(link.attr(sentry, ga::kWeakened).value_or(1)) == 0 && ready_yaw &&
               ready_pitch && std::abs(*ready_pitch) <= kTakeoverProbePitchEnvelopeDegrees;
      },
      30, 0.1);
    if (!actuator_ready) {
      tools::logger()->error(
        "[gestalt] {} takeover gate failed: actuator not physically ready map={} "
        "hp={:.0f} defeated={:.0f} can_operate={:.0f} chassis_online={:.0f} "
        "weakened={:.0f} yaw={:.2f} pitch={:.2f}",
        phase, sentry, link.attr(sentry, ga::kHealth).value_or(-1),
        link.attr(sentry, ga::kDefeated).value_or(-1),
        link.attr(sentry, ga::kCanOperate).value_or(-1),
        link.attr(sentry, ga::kIsChassisOnline).value_or(-1),
        link.attr(sentry, ga::kWeakened).value_or(-1), ready_yaw.value_or(0.0),
        ready_pitch.value_or(0.0));
      return false;
    }

    constexpr int kResponseAttempts = 2;
    constexpr double kRestoreToleranceDegrees = 1.5;
    constexpr int kRequiredRestoreSamples = 2;
    bool responded = false;
    bool restored = false;
    std::optional<double> yaw0, pitch0, response_yaw, response_pitch, restore_yaw,
      restore_pitch;
    double best_yaw_error = std::numeric_limits<double>::infinity();
    double best_pitch_error = std::numeric_limits<double>::infinity();
    double max_yaw_motion = 0.0;
    double max_pitch_motion = 0.0;
    int passed_attempt = 0;

    for (int attempt = 1; attempt <= kResponseAttempts && !responded; ++attempt) {
      yaw0.reset();
      pitch0.reset();
      if (!wait_for(
            [&] {
              yaw0 = link.attr(sentry, ga::kTurretYaw);
              pitch0 = link.attr(sentry, ga::kTurretPitch);
              return yaw0.has_value() && pitch0.has_value();
            },
            3, 0.1)) {
        tools::logger()->error("[gestalt] {} takeover gate failed: turret telemetry absent", phase);
        return false;
      }

      const double yaw_step = attempt % 2 == 1 ? 3.0 : -3.0;
      const double target_yaw = std::remainder(*yaw0 + yaw_step, 360.0);
      const double target_pitch =
        *pitch0 + (std::abs(*pitch0) < 1.0 ? 2.0 : (*pitch0 > 0 ? -2.0 : 2.0));
      const auto response_deadline = std::chrono::steady_clock::now() + 5s;
      while (std::chrono::steady_clock::now() < response_deadline) {
        link.exec(fmt::format(
          "UEExec RBExtAim {} {:.2f} {:.2f} 0", ext_pid, target_yaw, target_pitch));
        std::this_thread::sleep_for(100ms);
        response_yaw = link.attr(sentry, ga::kTurretYaw);
        response_pitch = link.attr(sentry, ga::kTurretPitch);
        if (!response_yaw || !response_pitch) continue;
        const double yaw_error = angle_error_degrees(*response_yaw, target_yaw);
        const double pitch_error = std::abs(*response_pitch - target_pitch);
        const double yaw_motion = angle_error_degrees(*response_yaw, *yaw0);
        const double pitch_motion = std::abs(*response_pitch - *pitch0);
        best_yaw_error = std::min(best_yaw_error, yaw_error);
        best_pitch_error = std::min(best_pitch_error, pitch_error);
        max_yaw_motion = std::max(max_yaw_motion, yaw_motion);
        max_pitch_motion = std::max(max_pitch_motion, pitch_motion);
        if (
          yaw_error <= 0.75 && pitch_error <= 0.75 && yaw_motion >= 1.5 &&
          pitch_motion >= 0.75) {
          responded = true;
          passed_attempt = attempt;
          break;
        }
      }

      // The probe is deliberately non-firing and restores each attempt's entry
      // pose before retrying or allowing normal control to continue. Reversing
      // a physical turret must first bleed commanded-direction momentum, so
      // require consecutive settled samples at the entry pose.
      const auto restore_deadline = std::chrono::steady_clock::now() + 5s;
      restored = false;
      int consecutive_restore_samples = 0;
      while (std::chrono::steady_clock::now() < restore_deadline) {
        link.exec(fmt::format("UEExec RBExtAim {} {:.2f} {:.2f} 0", ext_pid, *yaw0, *pitch0));
        std::this_thread::sleep_for(100ms);
        restore_yaw = link.attr(sentry, ga::kTurretYaw);
        restore_pitch = link.attr(sentry, ga::kTurretPitch);
        if (
          restore_yaw && restore_pitch &&
          angle_error_degrees(*restore_yaw, *yaw0) <= kRestoreToleranceDegrees &&
          std::abs(*restore_pitch - *pitch0) <= kRestoreToleranceDegrees) {
          if (++consecutive_restore_samples >= kRequiredRestoreSamples) {
            restored = true;
            break;
          }
        } else {
          consecutive_restore_samples = 0;
        }
      }
      if (responded || !restored) break;
      tools::logger()->warn(
        "[gestalt] {} takeover response attempt {}/{} did not reach target; retrying "
        "after physical settle",
        phase, attempt, kResponseAttempts);
      std::this_thread::sleep_for(1s);
    }

    if (!responded || !restored) {
      tools::logger()->error(
        "[gestalt] {} takeover gate failed: RBExtAim response={} pose_restore={} map={} "
        "response_yaw={:.2f} best_yaw_error={:.2f} yaw_motion={:.2f} "
        "response_pitch={:.2f} best_pitch_error={:.2f} pitch_motion={:.2f} "
        "restore_yaw={:.2f} restore_yaw_error={:.2f} restore_pitch={:.2f} "
        "restore_pitch_error={:.2f}",
        phase, responded, restored, sentry, response_yaw.value_or(0.0), best_yaw_error,
        max_yaw_motion, response_pitch.value_or(0.0), best_pitch_error, max_pitch_motion,
        restore_yaw.value_or(0.0),
        restore_yaw && yaw0 ? angle_error_degrees(*restore_yaw, *yaw0) : -1.0,
        restore_pitch.value_or(0.0),
        restore_pitch && pitch0 ? std::abs(*restore_pitch - *pitch0) : -1.0);
      return false;
    }
    tools::logger()->info(
      "[gestalt] {} takeover gate passed: mode90 + frame contract + physically-ready "
      "RBExtAim response (attempt {})",
      phase, passed_attempt);
    return true;
  };

  auto release_takeover = [&](const char * phase) {
    // Stop renewals before sending release; otherwise a heartbeat racing this
    // sequence can immediately re-acquire mode 90.
    claim_heartbeat.pause();
    const auto pid_map = link.find_player(ext_pid);
    const auto pid_health = pid_map ? link.attr(*pid_map, ga::kHealth) : std::nullopt;
    const int release_map = pid_map.value_or(-1);
    const bool terminal_inactive =
      bench_match && terminal_inactive_release_allowed &&
      link.attr(sentry, ga::kHealth).value_or(0) <= 0 &&
      (!pid_map || (pid_health && *pid_health <= 0));
    const bool expected_map = terminal_inactive || (pid_map && *pid_map == sentry);
    const bool expected_identity =
      !bench_match || terminal_inactive || (pid_map && is_expected_match_sentry(*pid_map));
    const double yaw = link.attr(release_map, ga::kTurretYaw).value_or(0.0);
    const double pitch = link.attr(release_map, ga::kTurretPitch).value_or(0.0);
    link.exec(fmt::format("UEExec RBExtAim {} {:.2f} {:.2f} 0", ext_pid, yaw, pitch));
    link.exec(fmt::format("ExtAimClaim {} 0", ext_pid));
    link.exec("UEExec RBTakeOver release");
    int observed_release_map = release_map;
    const bool observed_mode_restore = wait_for(
      [&] {
        const auto current = link.find_player(ext_pid);
        if (!current) return false;
        observed_release_map = *current;
        return link.attr(*current, ga::kTargetMode).value_or(90) != 90;
      },
      5, 0.1);
    const bool release_identity =
      !bench_match || terminal_inactive ||
      (observed_release_map > 0 && is_expected_match_sentry(observed_release_map));
    // With no live target at natural settlement, explicit claim/release was
    // sent and the heartbeat has remained stopped throughout the 5s validation
    // window above. A recycled map may retain stale mode=90 in this client's
    // cache, but it is no longer a controllable object.
    const bool restored = observed_mode_restore || terminal_inactive;
    const bool release_ok = restored && expected_map && expected_identity && release_identity;
    if (release_ok)
      tools::logger()->info(
        "[gestalt] {} release gate passed: pid={} current_map={} AITargetMode={} "
        "terminal_inactive={}", phase,
        ext_pid, observed_release_map,
        link.attr(observed_release_map, ga::kTargetMode).value_or(-1), terminal_inactive);
    else
      tools::logger()->error(
        "[gestalt] {} release gate failed: controlled_map={} pid_current_map={} "
        "map_match={} identity={} restored={} AITargetMode={} terminal_inactive={}",
        phase, sentry, observed_release_map, expected_map,
        expected_identity && release_identity, restored,
        link.attr(observed_release_map, ga::kTargetMode).value_or(-1), terminal_inactive);
    return release_ok;
  };

  if (bench_match) {
    // ---------- LIVE-MATCH takeover ----------
    // Ordering per the acceptance spec: manually SPAWN our sentry in the prep
    // stage, CLAIM its turret/fire, arm the camera/bridge, and only THEN start
    // the match — the vehicle is externally owned from second zero (claiming a
    // roster sentry after the whistle landed mid-fight: hp already 340/260 by
    // claim time, and the roster pawn's camera came up in TPS, which
    // invalidates the fov-25 first-person intrinsics; a pid-0 Respawn car uses
    // the same first-person takeover view as the bench). Chassis and economy
    // stay with the built-in AI after the claim.
    if (!wait_for([&] { return link.match_status() >= 0; }, 60)) {
      tools::logger()->error("[gestalt] match status unavailable during prep");
      return 2;
    }
    if (link.match_status() >= 1) {
      tools::logger()->error(
        "[gestalt] match already running; refusing late claim because pre-match gates cannot be "
        "proven");
      return 2;
    }
    const int prev_map = link.find_player(0).value_or(-1);
    bool spawned = false;
    for (int attempt = 0; attempt < 3 && !spawned; ++attempt) {
      link.exec(fmt::format("Respawn 0 {} {}", bench_shooter_entity, bench_shooter_team));
      spawned = wait_for(
        [&] {
          auto cur = link.find_player(0);
          return cur && *cur > prev_map && link.attr(*cur, ga::kHealth).value_or(0) > 0 &&
                 is_expected_match_sentry(*cur);
        },
        15);
    }
    if (!spawned) {
      tools::logger()->error("[gestalt] match sentry spawn failed after retries");
      return 1;
    }
    sentry = *link.find_player(0);
    if (!is_expected_match_sentry(sentry)) {
      tools::logger()->error(
        "[gestalt] spawned pid 0 failed red HACHISEN identity gate: map={} class={} team={} "
        "entity_config={}",
        sentry, link.attr(sentry, ga::kClass).value_or(-1),
        link.attr(sentry, ga::kTeamId).value_or(-1), link.player_entity_config(0).value_or(-1));
      return 2;
    }
    ext_pid = 0;
    outpost = link.find_red_outpost().value_or(-1);
    link.exec(fmt::format("SetAttribute 0 {} 1", ga::kIsAI));
    // ExtAimClaim atomically saves the strategy-owned target mode and switches
    // the current pawn to mode 90. Never pre-write 90 here: doing so destroys
    // the value the game must restore on release.
    claim_heartbeat.activate(ext_pid);
    link.exec("UEExec RBTakeOver 0");  // pid-0 pawn: same first-person gun view as the bench
    // One-time starting ammo grant (Respawn leaves the player allowance at 0);
    // in-match resupply afterwards is the built-in AI's business.
    link.exec(fmt::format("SetAttribute 0 {} {}", ga::kAllowance17mm, bench_allowance));
    link.exec("UEExec r.MotionBlurQuality 0");
    link.exec("UEExec r.AntiAliasingMethod 1");
    link.exec("UEExec r.RobotNav.DebugDraw 0");
    link.exec("UEExec t.MaxFPS 30");
    link.exec("UEExec r.VisionBridge.Enable 1");
    if (!ensure_frame_stream()) {
      release_takeover("pre-match-init-failure");
      return 2;
    }
    link.apply_camera(camera_json);
    if (!verify_takeover_contract("pre-match")) {
      release_takeover("pre-match-gate-failure");
      return 2;
    }
    if (link.match_status() < 1) {
      tools::logger()->info("[gestalt] armed — starting the match with pid 0 already claimed");
      link.exec("SetMatchStatus 1");
      if (!wait_for([&] { return link.match_status() >= 1; }, 30)) {
        tools::logger()->error("[gestalt] SetMatchStatus 1 had no observable effect");
        release_takeover("match-start-failure");
        return 2;
      }
    }
    tools::logger()->info(
      "[gestalt] match takeover: sentry map={} pid=0 team={} hp={} match_status={}", sentry,
      bench_shooter_team, link.attr(sentry, ga::kHealth).value_or(-1), link.match_status());
  } else {
  // ---------- prep-stage setup (mirrors tools/outpost_benchmark.py) ----------
  if (do_setup) {
    // Boot-readiness: the TS console isn't instantly live after the WS opens —
    // a too-early Respawn is silently dropped (observed: full run with no
    // sentry/outpost). Wait for telemetry, then retry the spawn.
    wait_for([&] { return link.match_status() >= 0; }, 60);
    const int previous_sentry = link.find_player(0).value_or(-1);
    bool spawned = false;
    for (int attempt = 0; attempt < 3 && !spawned; ++attempt) {
      // Shooter chassis comes from the config (Mario 66000008 default; sentry
      // bench uses 66000012). Team must be set at spawn time so armor
      // emissive color and detector class are correct.
      link.exec(fmt::format("Respawn 0 {} {}", bench_shooter_entity, bench_shooter_team));
      // Respawn is asynchronous.  An older alive sentry may remain in the
      // attribute cache, so merely finding class/team can return immediately
      // with stale map telemetry.  Wait for the newly allocated (higher) map.
      spawned = wait_for(
        [&] {
          auto current = link.find_player(0);
          return current && *current > previous_sentry;
        },
        15);
    }
    if (!spawned) {
      tools::logger()->error("[gestalt] sentry spawn failed after retries");
      return 1;
    }
  }
  auto sentry_opt = link.find_player(0);
  if (!sentry_opt) {
    tools::logger()->error("[gestalt] no live pid-0 vehicle in telemetry");
    return 1;
  }
  sentry = *sentry_opt;
  if (
    static_cast<int>(link.attr(sentry, ga::kClass).value_or(-1)) != bench_shooter_class ||
    static_cast<int>(link.attr(sentry, ga::kTeamId).value_or(-1)) != bench_shooter_team) {
    tools::logger()->error(
      "[gestalt] pid-0 vehicle is not the expected shooter (map={} class={} want={} team={})",
      sentry, link.attr(sentry, ga::kClass).value_or(-1), bench_shooter_class,
      bench_shooter_team);
    return 1;
  }
  double boot_bearing_deg = -83.3;  // outpost-lob point → red outpost, map-4 fallback
  wait_for(
    [&] {
      auto op = link.find_red_outpost();
      return op && link.attr(*op, ga::kHealth).value_or(0) > 0;
    },
    30);
  outpost = link.find_red_outpost().value_or(-1);
  tools::logger()->info(
    "[gestalt] sentry map={} outpost map={} outpost_hp={}", sentry, outpost,
    outpost > 0 ? link.attr(outpost, ga::kHealth).value_or(-1) : -1);

  if (do_setup) {
    // The benchmark harness controls pid 0 only. Never use a team/class BatchSet
    // here: the standard scene deliberately contains a blue infantry target at
    // the fortress, and a broad write would silently take control of that target.
    link.exec(fmt::format("SetAttribute 0 {} 1", ga::kIsAI));
    // Exact marker drive is selected by TargetMode=0. The bridge must not add
    // turret scan/levelling policy while the vehicle navigates.
    link.exec(fmt::format("SetAttribute 0 {} 0", ga::kTargetMode));
    bool arrived = false;
    {
      // Canonical outpost-lob observer point: Blue GMapInfo mode-5 destination
      // at (-4.50, 3.00)m.  A blue range hero's BASE-lob point is NOT generic
      // Marker_40: the match strategy drives AIMoveMode=6 to the exact
      // GMapInfo.DeploymentBlue point (-5.20, 8.50)m.  Marker_40 is only the
      // broad deployment zone and its center may be off the usable navmesh.
      constexpr double kOutpostLobXcm = -450.0;
      constexpr double kOutpostLobYcm = 300.0;
      link.exec(fmt::format("SetAttribute 0 {} 5", ga::kMoveMode));
      auto t_drive_end =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(120);
      int tick = 0;
      while (std::chrono::steady_clock::now() < t_drive_end) {
        auto x = link.attr(sentry, ga::kPosX), y = link.attr(sentry, ga::kPosY);
        if (tick % 2 == 0 && x && y) {
          tools::logger()->info(
            "[gestalt] outpost-lob drive pos=({:.2f},{:.2f})m goal=(-4.50,3.00)m "
            "error={:.2f}m",
            *x / 100.0, *y / 100.0,
            std::hypot(*x - kOutpostLobXcm, *y - kOutpostLobYcm) / 100.0);
        }
        if (
          x && y &&
          std::hypot(*x - kOutpostLobXcm, *y - kOutpostLobYcm) < 70.0) {
          arrived = true;
          break;
        }
        ++tick;
        std::this_thread::sleep_for(500ms);
      }
    }
    tools::logger()->info("[gestalt] pos={} drive arrived={}", pos, arrived);
    if (!arrived) {
      tools::logger()->error("[gestalt] invalid bench: sentry did not reach requested marker");
      return 2;
    }
    link.exec(fmt::format("SetAttribute 0 {} 2", ga::kMoveMode));      // hold
    // Claim owns the mode-90 transition and remembers the prior mode for
    // release; do not pre-write target mode here.
    claim_heartbeat.activate(ext_pid);
    link.exec("UEExec RBTakeOver 0");  // guarantee the viewport IS the gun view
    // Respawn does NOT refill the 17mm allowance (player-scoped) — reset it so
    // every run starts with exactly the configured round budget.
    link.exec(fmt::format("SetAttribute 0 {} {}", ga::kAllowance17mm, bench_allowance));
    // Motion blur smears the lightbars during turret slew (shutterSpeed only
    // drives exposure, not blur) — kill it for detection stability.
    link.exec("UEExec r.MotionBlurQuality 0");
    link.exec("UEExec r.AntiAliasingMethod 1");  // FXAA: measured 99.7% detector recall
    link.exec("UEExec r.RobotNav.DebugDraw 0");  // path overlays contaminate armor pixels
    // Cap the game render at 30fps: frees CPU cores for OpenVINO inference
    // (60fps rendering starves the detector to ~110ms/frame, tripping the
    // tracker's hardcoded dt>0.1s lost-guard). 30fps present rate still
    // exceeds the vision loop rate.
    link.exec("UEExec t.MaxFPS 30");
    link.exec("UEExec r.VisionBridge.Enable 1");
    std::this_thread::sleep_for(2s);
    link.apply_camera(camera_json);
    std::this_thread::sleep_for(1500ms);

    // ---- bench target (vehicle rounds): spawn → drive → arrive → boost → spin.
    // The target is an RBNavLab car (own per-tick mode maintenance, not in the
    // AiPlayerService table) so nothing here fights the match strategy.
    if (bench_vehicle) {
      // Idempotent rerun: if a LIVE car already owns this pid (earlier attempt
      // left it standing), reuse it — re-spawning an occupied pid never yields
      // the new map the guard below waits for.
      const auto pre = link.find_player(bench_target_pid);
      const bool target_alive = pre && link.attr(*pre, ga::kHealth).value_or(0) > 0;
      if (!bench_target_spawn.empty() && !target_alive) {
        // Same stale-corpse guard as the shooter spawn: a torn-down target from
        // an earlier run leaves a FROZEN dead map under this pid — only accept
        // a newly allocated (higher) map id.
        const int prev_target = pre.value_or(-1);
        link.exec(bench_target_spawn);
        if (!wait_for(
              [&] {
                auto current = link.find_player(bench_target_pid);
                return current && *current > prev_target &&
                       link.attr(*current, ga::kHealth).value_or(0) > 0;
              },
              20)) {
          tools::logger()->error(
            "[gestalt] bench target pid={} did not appear as a NEW live map", bench_target_pid);
          return 1;
        }
      }
      const int tmap = link.find_player(bench_target_pid).value_or(-1);
      if (tmap < 0) {
        tools::logger()->error("[gestalt] no live bench target pid={}", bench_target_pid);
        return 1;
      }
      if (!bench_target_drive.empty()) {
        link.exec(bench_target_drive);
        // Arrival: explicit goal → the same 70cm ball as the shooter drive;
        // mode-style drives (fortress/deployment) have no coordinate here, so
        // wait for the nav to settle (three consecutive 2s samples within 15cm).
        auto t_tgt_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(150);
        double px = 1e9, py = 1e9;
        int still = 0;
        bool tgt_arrived = false;
        while (std::chrono::steady_clock::now() < t_tgt_end) {
          std::this_thread::sleep_for(2s);
          auto x = link.attr(tmap, ga::kPosX), y = link.attr(tmap, ga::kPosY);
          if (!x || !y) continue;
          if (bench_target_goal_cm.size() >= 2) {
            const double err =
              std::hypot(*x - bench_target_goal_cm[0], *y - bench_target_goal_cm[1]);
            tools::logger()->info(
              "[gestalt] target drive pos=({:.2f},{:.2f})m error={:.2f}m", *x / 100.0,
              *y / 100.0, err / 100.0);
            if (err < bench_target_goal_radius_cm) {
              tgt_arrived = true;
              break;
            }
          } else {
            still = std::hypot(*x - px, *y - py) < 15.0 ? still + 1 : 0;
            px = *x;
            py = *y;
            if (still >= 3) {
              tgt_arrived = true;
              break;
            }
          }
        }
        if (!tgt_arrived) {
          tools::logger()->error("[gestalt] invalid bench: target never reached its post");
          return 2;
        }
      }
      if (bench_target_hp > 0) {
        // HealthMax clamps direct HP writes (observed: 5000 → stuck at the
        // stock 150) — raise the ceiling first, then the health.
        link.exec(fmt::format("RBNavLab set {} {} {}", bench_target_pid, ga::kHealthMax,
                              bench_target_hp));
        link.exec(fmt::format("RBNavLab set {} {} {}", bench_target_pid, ga::kHealth,
                              bench_target_hp));
      }
      if (!bench_target_spin.empty()) link.exec(bench_target_spin);
      std::this_thread::sleep_for(1s);
      tools::logger()->info(
        "[gestalt] bench target ready map={} pos=({:.2f},{:.2f})m hp={}", tmap,
        link.attr(tmap, ga::kPosX).value_or(0) / 100.0,
        link.attr(tmap, ga::kPosY).value_or(0) / 100.0,
        link.attr(tmap, ga::kHealth).value_or(-1));
    }

    // Bootstrap: face the outpost so plates enter the frame (GT used once,
    // like a driver would park the robot facing the target).
    auto sx = link.attr(sentry, ga::kPosX).value_or(0) / 100.0;
    auto sy = link.attr(sentry, ga::kPosY).value_or(0) / 100.0;
    if (bench_vehicle) {
      // Same GT-used-once convention as the outpost park: the target's spawn
      // position seeds the boot bearing, then never enters the aim path.
      const int tmap = link.find_player(bench_target_pid).value_or(-1);
      const double txm = tmap > 0 ? link.attr(tmap, ga::kPosX).value_or(0) / 100.0 : 0;
      const double tym = tmap > 0 ? link.attr(tmap, ga::kPosY).value_or(0) / 100.0 : 0;
      boot_bearing_deg = std::atan2(tym - sy, txm - sx) * kRad2Deg;
    } else {
      boot_bearing_deg = std::atan2(kRedOutpostY - sy, kRedOutpostX - sx) * kRad2Deg;
    }
    bool boot_aligned = false;
    const auto t_align_end = std::chrono::steady_clock::now() + 8s;
    while (std::chrono::steady_clock::now() < t_align_end) {
      link.exec(
        fmt::format("UEExec RBExtAim 0 {:.1f} {:.1f} 0", boot_bearing_deg, bench_boot_pitch));
      const auto yaw = link.attr(sentry, ga::kTurretYaw);
      const auto pitch = link.attr(sentry, ga::kTurretPitch);
      if (
        yaw && pitch && std::abs(std::remainder(*yaw - boot_bearing_deg, 360.0)) < 1.5 &&
        std::abs(*pitch - bench_boot_pitch) < 1.5) {
        boot_aligned = true;
        break;
      }
      std::this_thread::sleep_for(250ms);
    }
    if (!boot_aligned) {
      tools::logger()->error(
        "[gestalt] invalid bench: turret did not settle at outpost bootstrap pose");
      return 2;
    }
  }
  else {
    // A no-setup diagnostic must still apply the camera/render half of the
    // selected configuration.  Otherwise switching from FOV25 to FOV15 while
    // reusing an existing vehicle silently leaves K and the rendered FOV out
    // of sync, invalidating every PnP/reprojection comparison.
    link.exec("UEExec r.MotionBlurQuality 0");
    link.exec("UEExec r.AntiAliasingMethod 1");
    link.exec("UEExec r.RobotNav.DebugDraw 0");
    link.exec("UEExec t.MaxFPS 30");
    link.exec("UEExec r.VisionBridge.Enable 1");
    link.apply_camera(camera_json);
    std::this_thread::sleep_for(1500ms);
  }
  }  // !bench_match (bench setup)

  // HP probe: the entity whose health decides hits/exit. Vehicle rounds read
  // the RBNavLab target's map; the classic bench keeps the red outpost; match
  // takeover watches OUR OWN sentry (exit on its death = survival semantics).
  int probe = bench_match ? sentry : outpost;
  if (bench_vehicle) {
    const auto tmap = link.find_player(bench_target_pid);
    if (!tmap) {
      tools::logger()->error("[gestalt] bench target pid={} not in telemetry", bench_target_pid);
      return 1;
    }
    probe = *tmap;
  }

  // ---------- capture + the ORIGINAL auto_aim chain ----------
  // The no-setup diagnostic historically acquired its claim on the first loop
  // heartbeat. Start the independent lease here instead; setup/match paths are
  // idempotent renewals of the claim they already established.
  claim_heartbeat.activate(ext_pid);
  if (!ensure_frame_stream()) return 1;

  // Sim render calibration: the UE lightbars are drawn INSET vs the physical
  // 135x56mm armor model — raw PnP over-solves range by ~1.192x (measured at a
  // GT-anchored 4.28m standoff; see tools/aim_solver.py PER_NAME_QUAD_SCALE).
  // Expanding the detected quad about its centroid by the same factor feeds the
  // UNMODIFIED solver keypoints consistent with its full-size object model:
  // bearing unchanged, range (and thus ballistic pitch + fly-time) corrected.
  tools::logger()->info(
    "[gestalt] sim_quad_scale={} z_adapter={} bar_refine={} v_adapter={} delay={}", sim_quad_scale,
    z_adapter, bar_refine, v_adapter, cfg_delay);
  std::deque<double> v_hist_x, v_hist_y;  // vehicle plate-position history (v_adapter)
  std::deque<double> va_vlat;             // recent lateral velocities (rotation evidence)
  std::deque<double> va_dest;             // width-ratio range estimates (median-filtered)
  double va_prev_lat = 0, va_prev_t = -1;

  // 2026-outpost z-stagger adapter (bridge layer; sp_vision_25's EKF locks all
  // 3 plates to ONE height, but the 2026 drum staggers them ~101/111/142 cm).
  // Learn each plate's true z by binning raw measurements by drum phase
  // relative to the EKF tracked-plate angle, then correct the commanded pitch
  // for the plate the aimer chose.
  double zsum[3] = {0, 0, 0};
  int zn[3] = {0, 0, 0};
  // un-stagger state: online 3-cluster of raw plate heights + last EKF pose
  double zc_mean[3] = {0, 0, 0};
  int zc_cnt[3] = {0, 0, 0};
  int zc_n = 0;
  std::deque<double> ctr_hist_x, ctr_hist_y;  // drum-center running estimate
  // Bridge phase filter: rulebook |omega|=2.513 (same prior sp_vision hardcodes
  // in target.cpp), sign locked by majority vote of measured phase deltas,
  // measurements folded mod 120° onto the nearest plate slot (single-frame
  // radial phase is noisy: ~0.5-1° bearing noise vs the 0.28m orbit).
  double ph_hat = 0, ph_t = -1;
  int ph_sign = 0;  // 0 = unlocked
  bool ph_geometry_frozen = false;
  double ph_cx = 0, ph_cy = 0, ph_zmid = 0;
  std::deque<int> ph_votes;
  constexpr double kOmegaBook = 2.513;
  // Per-height visibility tracks for the SIGN.  The three staggered outpost
  // plates are distinguishable by height, but detections from different
  // heights are interleaved in the same frame.  A single episode buffer would
  // therefore splice three different plates together and make the sign vote a
  // coin flip.  Keep one unwrapped phase history per learned height cluster.
  std::vector<std::pair<double, double>> ep_samples[3];  // (t, phi_unwrapped)
  double ep_last_t[3] = {-1, -1, -1};
  double ep_last_phi[3] = {0, 0, 0};
  double ep_last_vote_t[3] = {-1, -1, -1};
  double ph_hyp_last_t = -1;
  double last_cx = 0, last_cy = 0, last_a = 0;
  bool last_target_ok = false;
  double cfg_fy = 1545.1;
  try {
    auto km = YAML::LoadFile(config)["camera_matrix"].as<std::vector<double>>();
    if (km.size() == 9) cfg_fy = km[4];
  } catch (...) {
  }
  auto wrap_rad = [](double a) {
    while (a > CV_PI) a -= 2 * CV_PI;
    while (a < -CV_PI) a += 2 * CV_PI;
    return a;
  };
  auto phase_bucket = [&](double rel_angle) {
    int k = static_cast<int>(std::lround(wrap_rad(rel_angle) / (2.0 * CV_PI / 3.0)));
    return ((k % 3) + 3) % 3;
  };

  const double hp0 = probe > 0 ? link.attr(probe, ga::kHealth).value_or(1500) : 1500;
  double allowance_base = link.attr(sentry, ga::kAllowance17mm).value_or(bench_allowance);
  // Baseline for the game's cumulative per-vehicle shot counter (63000002
  // BulletFiredTotal): "shots" must mean rounds that LEFT THE GUN, not trigger
  // requests — the loop re-asserts fire across consecutive frames while the
  // gun is rate/heat-limited, so command counts run absurdly high.
  double fired0 = link.attr(sentry, ga::kBulletsFired).value_or(0);
  double fired_carry = 0;  // rounds fired in previous lives (match respawns)
  // Combat effectiveness (match mode): cumulative damage dealt, per-life
  // baselined because a respawn starts a fresh combat map.
  double dealt_base = link.attr(sentry, ga::kDamageApplied).value_or(0);
  double last_dealt = dealt_base, dealt_carry = 0;
  bool current_life_accounted = false;
  int lives_lost = 0;
  double last_revive_wait_s = 0;
  bool takeover_contract_ok = true;
  bool observed_match_end = false;

  // Async debug window: imshow/waitKey stall 40-80ms on this box — feeding a
  // UI thread through a latest-frame mailbox keeps the vision loop under the
  // tracker's hardcoded dt>0.1s lost-guard while the window stays live.
  std::mutex ui_mu;
  cv::Mat ui_frame;
  std::atomic<bool> ui_quit{false};
  std::thread ui_thread([&] {
    cv::Mat local;
    while (!ui_quit) {
      {
        std::lock_guard<std::mutex> lk(ui_mu);
        if (!ui_frame.empty()) {
          ui_frame.copyTo(local);
          ui_frame.release();
        }
      }
      if (!local.empty()) {
        cv::resize(local, local, {}, 0.5, 0.5);
        cv::imshow("reprojection", local);
        local.release();
      }
      if (cv::waitKey(30) == 'q') ui_quit = true;
    }
  });

  io::Command last_cmd{false, false, 0, 0};
  // Match-mode idle scan: sweep the turret when the aimer has produced no
  // solution for a while, freezing whenever raw detections appear. Yaw spins
  // full circles while pitch rides a ±15° triangle wave — low targets under
  // the muzzle line are invisible to a level-only sweep. The two periods are
  // deliberately non-integer-ratio (yaw rev 6s vs pitch leg ~8.6s) so
  // successive revolutions cover different elevation bands (spiral coverage).
  constexpr double kScanAfterLostSec = 2.0, kScanRateDegS = 60.0;
  constexpr double kScanPitchMaxDeg = 15.0, kScanPitchRateDegS = 3.5;
  // Integrate against wall-clock time rather than assuming a render/detector
  // rate. A long capture/inference stall is capped so recovery cannot issue a
  // large one-frame aim jump.
  constexpr double kScanMaxStepSec = 0.20;
  double scan_yaw_deg = 0.0, scan_pitch_deg = 0.0;
  int scan_pitch_dir = -1;  // start by looking down: close targets hide there
  bool scan_active = false;
  auto t_last_ctrl = std::chrono::steady_clock::now();
  auto t_last_scan_step = t_last_ctrl;
  int frames = 0, det_frames = 0, cmds = 0, fire_cmds = 0, grab_fail = 0;
  bool last_fire_sent = false;
  // RBExtAim carries a latched input bit in the game. A lost target, capture
  // failure, death, or shutdown must actively send fire=0; merely stopping
  // RBExtAim commands (or releasing ExtAimClaim) does not clear that bit.
  auto release_fire = [&](bool force = false) {
    if (!force && !last_fire_sent) return;
    const double yaw = link.attr(sentry, ga::kTurretYaw).value_or(0.0);
    const double pitch = link.attr(sentry, ga::kTurretPitch).value_or(0.0);
    link.exec(fmt::format("UEExec RBExtAim {} {:.2f} {:.2f} 0", ext_pid, yaw, pitch));
    last_fire_sent = false;
    cmds++;
  };
  uint64_t ws_generation = link.connection_generation();
  bool ws_continuity_clean = true;
  bool frame_writer_contract_ok = cap.process_id() != 0;
  bool capture_recovery_pending = false;
  auto rearm_takeover = [&](const char * phase, bool reinitialize_capture) {
    if (!wait_for([&] { return link.connected(); }, 20, 0.1)) {
      tools::logger()->error("[gestalt] {} re-arm failed: WS is not connected", phase);
      return false;
    }
    std::optional<int> current;
    if (!wait_for(
          [&] {
            current = link.find_player(ext_pid);
            return current && *current == sentry &&
              link.attr(*current, ga::kHealth).value_or(0) > 0;
          },
          15, 0.1)) {
      tools::logger()->error(
        "[gestalt] {} re-arm failed: controlled={} pid_current={} is not the expected live map",
        phase, sentry, current.value_or(-1));
      return false;
    }
    if (bench_match && !is_expected_match_sentry(*current)) {
      identity_contract_ok = false;
      tools::logger()->error(
        "[gestalt] {} re-arm failed red HACHISEN identity: map={} class={} team={} config={}",
        phase, *current, link.attr(*current, ga::kClass).value_or(-1),
        link.attr(*current, ga::kTeamId).value_or(-1),
        link.player_entity_config(ext_pid).value_or(-1));
      return false;
    }
    claim_heartbeat.activate(ext_pid);
    link.exec(fmt::format("UEExec RBTakeOver {}", ext_pid));
    link.exec("UEExec r.VisionBridge.Enable 1");
    link.exec("UEExec r.MotionBlurQuality 0");
    link.exec("UEExec r.AntiAliasingMethod 1");
    link.exec("UEExec r.RobotNav.DebugDraw 0");
    link.exec("UEExec t.MaxFPS 30");
    if (reinitialize_capture) cap_initialized = false;
    if (!ensure_frame_stream()) return false;
    link.apply_camera(camera_json);
    if (!verify_takeover_contract(phase)) return false;
    frame_writer_contract_ok =
      frame_writer_contract_ok && cap.process_id() != 0;
    return true;
  };
  // Per-hit damage calibration: histogram of discrete HP drops observed on the
  // probe (logged at RESULT so bench_damage_per_hit can be verified per target).
  std::map<int, int> hp_drop_hist;
  double hp_prev = -1;
  int det_color_hist[4] = {0, 0, 0, 0};  // red/blue/extinguish/purple raw counts
  auto t_ekf_log = std::chrono::steady_clock::now();

  // ---- BASIC mode state (fixed-axis median filter + known-omega timed fire) ----
  // The turret HOLDS the drum-axis bearing (zero command jitter — the ideal
  // LQR otherwise reproduces every aimer flip-flop), pitch pre-positions for
  // the NEXT-arriving plate's learned height, and fire is scheduled so the
  // bullet ARRIVES at the plate crossing: t_fire = t_cross - fly - delay.
  std::deque<double> b_yaw_hist, b_dist_hist, b_z_hist;
  double b_phase_anchor = 0, b_phase_anchor_t = -1;  // de-rotated plate phase base
  double b_omega = 0;                                // signed rad/s (game truth)
  double b_plate_z[3] = {0, 0, 0};
  int b_plate_zn[3] = {0, 0, 0};
  FILE * rec_fp = nullptr;
  if (do_record) rec_fp = std::fopen("gestalt_record.jsonl", "w");

  // ---- OBSERVE mode: camera-parameter sweep bench (no aiming, no firing) ----
  struct SweepPhase
  {
    std::string name, camera, exec;
    double secs = 18;
  };
  std::vector<SweepPhase> sweep;
  if (mode_observe) {
    try {
      auto ysweep = YAML::LoadFile(config)["sweep"];
      for (const auto & ph : ysweep) {
        SweepPhase p;
        p.name = ph["name"].as<std::string>();
        if (ph["camera"].IsDefined()) p.camera = ph["camera"].as<std::string>();
        if (ph["exec"].IsDefined()) p.exec = ph["exec"].as<std::string>();
        if (ph["secs"].IsDefined()) p.secs = ph["secs"].as<double>();
        sweep.push_back(p);
      }
    } catch (...) {
    }
    tools::logger()->info("[observe] sweep phases={}", sweep.size());
  }
  size_t sweep_idx = 0;
  bool sweep_phase_started = false;
  auto sweep_t0 = std::chrono::steady_clock::now();
  int sw_frames = 0, sw_det = 0;
  double sw_conf = 0, sw_width = 0, sw_last_w = 0;
  std::string sw_last_state = "-";
  auto median_of = [](std::deque<double> & d) {
    std::vector<double> v(d.begin(), d.end());
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  };
  auto launch_pitch = [](double v0, double d, double h) -> std::pair<double, double> {
    // closed-form low arc (mirrors tools/trajectory.cpp), returns (pitch, fly)
    double a = 9.81 * d * d / (2 * v0 * v0), b = -d, c = a + h;
    double delta = b * b - 4 * a * c;
    if (delta < 0) return {0.2, d / v0};
    double p1 = std::atan((-b + std::sqrt(delta)) / (2 * a));
    double p2 = std::atan((-b - std::sqrt(delta)) / (2 * a));
    double t1 = d / (v0 * std::cos(p1)), t2 = d / (v0 * std::cos(p2));
    return t1 < t2 ? std::make_pair(p1, t1) : std::make_pair(p2, t2);
  };
  auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  auto t_log = std::chrono::steady_clock::now();

  bool hp_seen = false;
  while (std::chrono::steady_clock::now() < t_end) {
    // Read connected first and generation second. connect_once publishes the
    // generation before connected=true, so this order cannot observe a new
    // connection paired with the old generation and skip the full re-arm.
    const bool ws_connected = link.connected();
    const auto current_generation = link.connection_generation();
    if (!ws_connected || current_generation != ws_generation) {
      ws_continuity_clean = false;
      release_fire(true);
      claim_heartbeat.pause();
      tools::logger()->warn(
        "[gestalt] WS continuity lost: generation {} -> {}; pausing claim until full re-arm",
        ws_generation, current_generation);
      if (!wait_for([&] { return link.connected(); }, 20, 0.1)) {
        takeover_contract_ok = false;
        tools::logger()->error("[gestalt] WS did not reconnect within 20s; failing safely");
        break;
      }
      if (!wait_for([&] { return link.match_status() >= 0; }, 15, 0.1)) {
        takeover_contract_ok = false;
        tools::logger()->error("[gestalt] WS reconnected without fresh match telemetry");
        break;
      }
      if (bench_match && link.match_status() >= 2) {
        observed_match_end = true;
        break;
      }
      if (!rearm_takeover("ws-recovery", true)) {
        takeover_contract_ok = false;
        tools::logger()->error("[gestalt] WS recovery failed full takeover gate");
        break;
      }
      ws_generation = link.connection_generation();
      ws_continuity_clean = true;
      capture_recovery_pending = false;
      grab_fail = 0;
      continue;
    }
    const double hp = probe > 0 ? link.attr(probe, ga::kHealth).value_or(0) : -1;
    const double allow = link.attr(sentry, ga::kAllowance17mm).value_or(0);
    // Actual rounds fired so far (BulletFiredTotal delta; allowance drain is
    // the fallback if the counter attribute is not in this build's telemetry).
    const auto bf_now = link.attr(sentry, ga::kBulletsFired);
    const double fired =
      fired_carry +
      (bf_now ? std::max(0.0, *bf_now - fired0)
              : std::max(0.0, allowance_base - allow));
    if (const auto dd = link.attr(sentry, ga::kDamageApplied)) last_dealt = *dd;
    if (hp_prev > 0 && hp > 0 && hp < hp_prev)
      hp_drop_hist[static_cast<int>(std::lround(hp_prev - hp))]++;
    if (hp > 0) hp_prev = hp;
    if (hp > 0) hp_seen = true;
    // Only honor hp==0 after a live reading: at boot the attribute may simply
    // not have arrived yet (value_or(0) — the 0.12-sweep instant-exit race).
    if (hp_seen && hp == 0) {
      release_fire();
      if (!bench_match) break;
      // Match takeover: our sentry died — it revives (buy-revive/auto), so
      // re-acquire the revived combat map and claim it for the next life.
      lives_lost++;
      dealt_carry += std::max(0.0, last_dealt - dealt_base);
      fired_carry = fired;
      current_life_accounted = true;
      // A dead pawn is no longer externally controlled. Stop renewing and
      // release explicitly so the game restores its saved strategy mode
      // without reporting the five-second safety lease as an expired claim
      // while we wait (which may take minutes) for the game-owned revive.
      claim_heartbeat.pause();
      link.exec(fmt::format("ExtAimClaim {} 0", ext_pid));
      tools::logger()->info(
        "[gestalt] own sentry down (life {} lost, dealt so far {:.0f}) — waiting for revive",
        lives_lost, dealt_carry);
      // Revival belongs to the GAME's own logic (sentry auto/buy-revive) — we
      // only park the vision loop, display the live revive progress, and wait
      // for the vehicle to come back on its own.
      const int dead_map = sentry;
      const auto t_dead0 = std::chrono::steady_clock::now();
      int new_map = -1;
      bool match_over = false;
      bool revive_identity_failed = false;
      while (!ui_quit && std::chrono::steady_clock::now() < t_end) {
        const double waited =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dead0).count();
        const bool death_ws_connected = link.connected();
        const auto death_ws_generation = link.connection_generation();
        if (!death_ws_connected || death_ws_generation != ws_generation) {
          ws_continuity_clean = false;
          claim_heartbeat.pause();
        }
        if (link.match_status() >= 2) {
          match_over = true;
          observed_match_end = true;
          terminal_inactive_release_allowed = true;
          break;
        }
        auto cur = link.find_player(0);
        // Game-owned revive has two valid forms: reconstruct a newer combat
        // map, or revive the same pawn/map in place. hp_seen already proved
        // this life reached zero, so HP>0 is a real post-death transition and
        // accepting the same map cannot mistake the old live snapshot for a
        // revive.
        if (cur && *cur >= dead_map && link.attr(*cur, ga::kHealth).value_or(0) > 0) {
          if (!is_expected_match_sentry(*cur)) {
            identity_contract_ok = false;
            revive_identity_failed = true;
            tools::logger()->error(
              "[gestalt] revived pid 0 is not red HACHISEN: map={} class={} team={} config={}",
              *cur, link.attr(*cur, ga::kClass).value_or(-1),
              link.attr(*cur, ga::kTeamId).value_or(-1),
              link.player_entity_config(0).value_or(-1));
          } else {
            new_map = *cur;
          }
          break;
        }
        const double rp = link.attr(dead_map, ga::kReviveProgress).value_or(0);
        const double rpm = link.attr(dead_map, ga::kReviveProgressMax).value_or(0);
        cv::Mat card(720, 1280, CV_8UC3, cv::Scalar(25, 25, 25));
        tools::draw_text(card, "SENTRY DOWN", {500, 320}, {0, 0, 255});
        tools::draw_text(
          card,
          fmt::format("revive {:.0f}/{:.0f}  waited {:.0f}s  lives_lost {}", rp, rpm, waited,
                      lives_lost),
          {380, 380}, {0, 255, 255});
        {
          std::lock_guard<std::mutex> lk(ui_mu);
          card.copyTo(ui_frame);
        }
        std::this_thread::sleep_for(500ms);
      }
      if (match_over || ui_quit || revive_identity_failed || new_map < 0) break;
      last_revive_wait_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dead0).count();
      sentry = new_map;
      probe = new_map;
      fired0 = link.attr(sentry, ga::kBulletsFired).value_or(0);
      allowance_base = link.attr(sentry, ga::kAllowance17mm).value_or(bench_allowance);
      dealt_base = link.attr(sentry, ga::kDamageApplied).value_or(0);
      last_dealt = dealt_base;
      current_life_accounted = false;
      hp_seen = false;
      hp_prev = -1;
      scan_active = false;
      // The pawn is revived (possibly in place): renew the claim, then
      // re-assert the view AND camera
      // rig. ExtAimClaim performs the mode-90 transition while preserving the
      // revived pawn's prior mode for release. A reconstructed pawn reverts to
      // the chassis preset
      // (Tower 400cm boom + FOV 90), which silently breaks the fov-25
      // intrinsics and gun-socket extrinsics for every life after the first
      // (observed: damage_dealt 377 → 15 across otherwise identical matches).
      const bool reconnect_during_death =
        !ws_continuity_clean || link.connection_generation() != ws_generation;
      if (!rearm_takeover("post-revive", reconnect_during_death)) {
        takeover_contract_ok = false;
        tools::logger()->error(
          "[gestalt] post-revive effect gate failed; refusing to resume control loop");
        break;
      }
      ws_generation = link.connection_generation();
      ws_continuity_clean = true;
      tools::logger()->info(
        "[gestalt] game revived our sentry after {:.0f}s: map={} (life {})",
        last_revive_wait_s, sentry, lives_lost + 1);
      continue;
    }
    if (!bench_match && hp_seen && allow <= 0) break;
    // Match takeover: economy belongs to the built-in AI (allowance may dip to
    // 0 until it re-buys), so only a settled match (or timeout) ends the run.
    if (bench_match && link.match_status() >= 2) {
      observed_match_end = true;
      break;
    }

    auto now = std::chrono::steady_clock::now();

    auto t_grab0 = std::chrono::steady_clock::now();
    auto frame = cap.grab(50);
    auto t_grab1 = std::chrono::steady_clock::now();
    static double grab_acc = 0, rej_acc = 0;
    static int grab_n = 0, rej_n = 0;
    if (!frame.ok) {
      rej_acc += std::chrono::duration<double, std::milli>(t_grab1 - t_grab0).count();
      rej_n++;
      ++grab_fail;
      if (grab_fail >= 10) capture_recovery_pending = true;
      if (grab_fail >= 100) {
        release_fire(true);
        if (!rearm_takeover("capture-recovery", true)) {
          takeover_contract_ok = false;
          tools::logger()->error(
            "[gestalt] capture did not recover through the full takeover gate");
          break;
        }
        capture_recovery_pending = false;
        grab_fail = 0;
        continue;
      }
      release_fire();
      continue;
    }
    grab_acc += std::chrono::duration<double, std::milli>(t_grab1 - t_grab0).count();
    grab_n++;
    if (frame.writer_process_id != cap.process_id()) {
      frame_writer_contract_ok = false;
      takeover_contract_ok = false;
      release_fire(true);
      tools::logger()->error(
        "[gestalt] frame writer pid mismatch: packet={} WS-bound={}", frame.writer_process_id,
        cap.process_id());
      break;
    }
    const bool frame_identity_ok =
      (frame.identity_flags & 0x7u) == 0x7u && frame.takeover_player_id == ext_pid &&
      frame.takeover_attribute_map_id == sentry && frame.view_actor_unique_id != 0 &&
      frame.view_actor_unique_id == frame.takeover_target_unique_id &&
      frame.takeover_epoch == verified_takeover_epoch;
    if (bench_match && !frame_identity_ok) {
      release_fire(true);
      tools::logger()->warn(
        "[gestalt] rendered-view identity changed: pid={} map={} view={} target={} epoch={} "
        "expected pid={} map={} epoch={} flags=0x{:X}; attempting full re-arm",
        frame.takeover_player_id, frame.takeover_attribute_map_id,
        frame.view_actor_unique_id, frame.takeover_target_unique_id, frame.takeover_epoch,
        ext_pid, sentry, verified_takeover_epoch, frame.identity_flags);
      if (!rearm_takeover("view-identity-recovery", false)) {
        takeover_contract_ok = false;
        frame_identity_contract_ok = false;
        tools::logger()->error("[gestalt] rendered-view identity recovery failed full gate");
        break;
      }
      capture_recovery_pending = false;
      grab_fail = 0;
      continue;
    }
    if (capture_recovery_pending) {
      release_fire(true);
      if (!rearm_takeover("capture-recovery", false)) {
        takeover_contract_ok = false;
        tools::logger()->error("[gestalt] capture recovery failed full takeover gate");
        break;
      }
      capture_recovery_pending = false;
      grab_fail = 0;
      continue;
    }
    grab_fail = 0;
    frames++;

    // The local frame packet contains the FINAL FSceneView quaternion that
    // rendered these exact pixels. It replaces the old 30Hz AttributeMap
    // arrival-time interpolation entirely.
    if (!frame.has_camera_pose) {
      release_fire();
      continue;
    }
    const auto & qv = frame.camera_quaternion_ue_xyzw;
    Eigen::Quaterniond q_camera_ue(qv[3], qv[0], qv[1], qv[2]);
    q_camera_ue.normalize();

    // UE world is left-handed (X forward, Y right, Z up). sp_vision world is
    // right-handed (X forward, Y left, Z up). OpenCV camera is x-right,
    // y-down, z-forward. Applying both reflections yields a proper rotation.
    const Eigen::Matrix3d S_ueworld_to_rh{
      {1, 0, 0},
      {0, -1, 0},
      {0, 0, 1}};
    const Eigen::Matrix3d R_cvcam_to_uecamera{
      {0, 0, 1},
      {1, 0, 0},
      {0, -1, 0}};
    const Eigen::Matrix3d R_camera2world =
      S_ueworld_to_rh * q_camera_ue.toRotationMatrix() * R_cvcam_to_uecamera;
    const Eigen::Matrix3d R_gimbal2world =
      R_camera2world * R_camera2gimbal_cfg.transpose();
    solver.set_R_gimbal2world(Eigen::Quaterniond(R_gimbal2world));

    const Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world, 2, 1, 0);
    const double tyaw = gimbal_ypr[0];
    const double tyaw_ue = -tyaw;
    const double tpitch = -gimbal_ypr[1];

    const auto t = qpc_to_steady(frame.t_present);
    auto t_det0 = std::chrono::steady_clock::now();
    auto armors = detector.detect(frame.img);
    auto t_det1 = std::chrono::steady_clock::now();
    if (!armors.empty()) det_frames++;
    // Detection histogram (raw, BEFORE tracker filters erase misclassified
    // plates) + color normalization: red outpost plates flicker to class-18
    // "blue outpost" at glancing angles; tracker.cpp:43 drops color!=enemy and
    // starves the EKF. The C++ tracker has no "enemy_color: any" — forcing the
    // color on outpost-named detections is the bridge-side equivalent.
    for (auto & armor : armors) {
      det_color_hist[static_cast<int>(armor.color) & 3]++;
      if (armor.name == auto_aim::ArmorName::outpost)
        armor.color = auto_aim::Color::red;
    }
    // Subpixel bar endpoints from raw pixels (falls back to yolo keypoints per armor)
    if (bar_refine)
      for (auto & armor : armors) refine_armor_points(frame.img, armor);
    // Snapshot for the debug window: yolo's own verdicts (conf/color/name/type),
    // as drawn by the original "detection" window (yolo11.cpp draw_detections),
    // captured before track() mutates/erases the list.
    auto armors_ui = armors;
    if (sim_quad_scale != 1.0) {
      for (auto & armor : armors) {
        cv::Point2f c(0.f, 0.f);
        for (const auto & p : armor.points) c += p;
        c *= 1.0f / static_cast<float>(armor.points.size());
        for (auto & p : armor.points)
          p = c + (p - c) * static_cast<float>(sim_quad_scale);
      }
    }
    // ---- 2026-drum MEASUREMENT SUBSTITUTION (bridge layer, before track) ----
    // Two structural mismatches between sp_vision_25's model and this render:
    //  (1) the 3 plates are height-STAGGERED (~101/111/142cm) while the EKF
    //      locks one ring height -> NIS resets;
    //  (2) the PnP plate NORMAL is frozen near the LOS at every resolution
    //      (bench: span 13-21° vs the true ±60° sweep) -> omega unobservable.
    // Both are fixed at the KEYPOINT boundary, pure vision, solver untouched:
    // solve each detection, estimate the drum center as the running mean of
    // plate positions (the orbit averages out), synthesize the plate pose
    // {x, y, z_mid, yaw = radial atan2(plate - center)}, and REPROJECT it with
    // the solver's own model into replacement keypoints. The tracker's internal
    // re-solve then recovers exactly this pose: flat drum + ROTATING normal.
    // ---- VEHICLE POSITION-SYNTHESIZED NORMAL (v_adapter, bridge layer) ----
    // Vehicle plates ride a ~0.25m ring around the chassis center; the plate's
    // radial azimuth about that center IS the rotation phase, measurable at
    // ~1-2deg/frame from POSITION alone (the per-frame PnP normal is unusable
    // at these plate scales -- yolo head flattens it and the flat-plate model
    // can't consume the 3D-bar asymmetry). Synthesize the plate normal as the
    // outward radial and reproject: the drum z_adapter recipe generalized to
    // 4-plate vehicles. Static targets (radial spread < 0.10m) pass through.
    if (v_adapter && !z_adapter) {
      static int va_applied = 0, va_rad_rej = 0, va_warm = 0, va_n = 0;
      static double va_rad_acc = 0;
      for (auto & armor : armors) solver.solve(armor);  // pre-solve: pair logic reads peers
      for (auto & armor : armors) {
        const double xr = armor.xyz_in_world[0], yr = armor.xyz_in_world[1];
        if (std::hypot(xr, yr) < 0.5) continue;
        v_hist_x.push_back(xr);
        v_hist_y.push_back(yr);
        if (v_hist_x.size() > 150) {
          v_hist_x.pop_front();
          v_hist_y.pop_front();
        }
        if (v_hist_x.size() < 30) {
          va_warm++;
          continue;
        }
        // median center: robust to the asymmetric per-plate sampling that
        // biases a mean and inflates/deflates the synthesized sweep
        const double cx = median_of(v_hist_x);
        const double cy = median_of(v_hist_y);
        // Rotation evidence gate on sustained LATERAL VELOCITY. Spatial spread
        // alone fails twice over: PnP depth noise on a static target looks like
        // radial motion, and a corner-aspect TWO-plate view hops between two
        // fixed lateral clusters (spread ~0.3m with zero rotation). A spinning
        // plate instead SWEEPS at |v_lat| = r*omega ~ 1.5 m/s with a constant
        // sign inside each visibility window, while switch jumps (>0.25m per
        // frame) break the chain.
        {
          const double c_norm = std::hypot(cx, cy);
          if (c_norm < 0.5) continue;
          const double px_ = -cy / c_norm, py_ = cx / c_norm;  // unit perpendicular to LOS
          const double lat = (xr - cx) * px_ + (yr - cy) * py_;
          const double t_now = frame.t_present;
          const double dt_lat = t_now - va_prev_t;
          if (va_prev_t > 0 && dt_lat > 0 && dt_lat < 0.12) {
            const double dlat = lat - va_prev_lat;
            if (std::abs(dlat) < 0.25)
              va_vlat.push_back(dlat / dt_lat);
            else
              va_vlat.clear();  // plate-switch jump
            while (va_vlat.size() > 3) va_vlat.pop_front();
          } else if (dt_lat >= 0.12) {
            va_vlat.clear();
          }
          va_prev_lat = lat;
          va_prev_t = t_now;
          bool sweeping = va_vlat.size() >= 2;
          for (size_t i = 0; sweeping && i < va_vlat.size(); ++i)
            sweeping = std::abs(va_vlat[i]) > 0.4 && (va_vlat[i] > 0) == (va_vlat.front() > 0);
          if (!sweeping) {
            // Static/ambiguous: the per-frame plate yaw is unobservable here and
            // optimize_yaw freezes at an arbitrary in-window angle (observed
            // near EDGE-ON, ~90deg off, drawing the model plate as a line).
            // Single plate: inject the FACE-ON prior (bounded error <=45deg by
            // the visibility cone). TWO plates in frame: their center-to-center
            // segment runs at 45deg to both faces, so the chassis orientation
            // is EXACT -- normals = segment azimuth +-45deg, pick per plate the
            // candidate nearest its face-on direction.
            const double beta = std::atan2(yr, xr);
            double yaw_inj = beta;  // face-on prior (single-plate case)
            double dist_corr = 1.0;
            if (armors.size() == 2) {
              // Two adjacent plates: tan(tilt) = W_other/W_this from apparent
              // widths (lateral, pixel-precise -- the center-to-center segment
              // azimuth is depth-noise dominated at this range and unusable).
              // Resolve the sign/side with a 4-combo bearing test against the
              // OTHER plate's observed bearing.
              const auto & other = (&armor == &armors.front()) ? armors.back() : armors.front();
              if (other.xyz_in_world.norm() > 0.5) {
                const double w_this = cv::norm(armor.points[0] - armor.points[1]);
                const double w_oth = cv::norm(other.points[0] - other.points[1]);
                if (w_this > 4 && w_oth > 4) {
                  const double phi = std::atan2(w_oth, w_this);  // |tilt| in (0,90)
                  // Width-ratio RANGE: W_this = m*cos(phi)*fy/d and cos(phi) =
                  // W_this/hypot(W1,W2)  =>  d = m*fy/hypot(W1,W2). Lateral
                  // pixel-precise, replaces the oblique-aspect PnP depth that
                  // runs ~0.7-0.9m long and inflates the drawn model boxes.
                  const double m_w = (armor.type == auto_aim::ArmorType::big) ? 0.230 : 0.135;
                  const double d_est = m_w * cfg_fy / std::hypot(w_this, w_oth);
                  if (d_est > 1.0 && d_est < 20.0) {
                    va_dest.push_back(d_est);
                    while (va_dest.size() > 20) va_dest.pop_front();
                  }
                  const double b_obs = std::atan2(other.xyz_in_world[1], other.xyz_in_world[0]);
                  const double s_half = 0.20;  // chassis half-size, m
                  double best_err = 1e9;
                  for (int cs = -1; cs <= 1; cs += 2) {
                    const double psi = wrap_rad(beta + cs * phi);
                    for (int side = -1; side <= 1; side += 2) {
                      const double qx = xr + s_half * (std::cos(psi) + std::cos(psi + side * CV_PI / 2));
                      const double qy = yr + s_half * (std::sin(psi) + std::sin(psi + side * CV_PI / 2));
                      const double err = std::abs(wrap_rad(std::atan2(qy, qx) - b_obs));
                      if (err < best_err) {
                        best_err = err;
                        yaw_inj = psi;
                      }
                    }
                  }
                }
              }
            }
            // Median-filtered width-ratio range: applied on EVERY static frame
            // (single- or two-plate) so the distance never toggles between the
            // corrected and raw-PnP value as the second plate flickers in and
            // out of detection (observed as r 0.20<->0.37 churn + NIS resets).
            if (va_dest.size() >= 5) {
              const double d_pnp = armor.xyz_in_world.norm();
              const double d_med = median_of(va_dest);
              if (d_pnp > 1.0) dist_corr = d_med / d_pnp;
            }
            auto pts_f = solver.reproject_armor(
              Eigen::Vector3d{xr * dist_corr, yr * dist_corr, armor.xyz_in_world[2] * dist_corr},
              yaw_inj, armor.type, armor.name);
            if (pts_f.size() == 4)
              for (int i = 0; i < 4; ++i) armor.points[i] = pts_f[i];
            continue;
          }
        }
        const double rad = std::hypot(xr - cx, yr - cy);
        va_rad_acc += rad;
        if (++va_n % 100 == 0)
          tools::logger()->info(
            "[vadapt] applied={} rad_rej={} warm={} rad_avg={:.3f}", va_applied, va_rad_rej,
            va_warm, va_rad_acc / va_n);
        if (rad < 0.06 || rad > 0.60) {
          va_rad_rej++;
          continue;  // static / implausible -> pass through
        }
        va_applied++;
        // armor-yaw convention: optimize_yaw searches +-70deg around the GIMBAL
        // heading, i.e. yaw points from camera INTO the scene -- the INWARD
        // ring normal (center-to-plate + pi), not the outward radial.
        const double phi = std::atan2(yr - cy, xr - cx) + CV_PI;
        auto pts = solver.reproject_armor(
          Eigen::Vector3d{xr, yr, armor.xyz_in_world[2]}, phi, armor.type, armor.name);
        if (pts.size() == 4)
          for (int i = 0; i < 4; ++i) armor.points[i] = pts[i];
      }
    }
    const bool phase_locked_at_frame_start = ph_sign != 0;
    if (z_adapter) {
      for (auto & armor : armors) {
        solver.solve(armor);  // raw world pose of THIS detection
        double zr = armor.xyz_in_world[2];
        double xr = armor.xyz_in_world[0], yr = armor.xyz_in_world[1];
        double dist = std::hypot(xr, yr);
        if (dist < 0.5) continue;
        ctr_hist_x.push_back(xr);
        ctr_hist_y.push_back(yr);
        if (ctr_hist_x.size() > 80) {
          ctr_hist_x.pop_front();
          ctr_hist_y.pop_front();
        }
        // learn per-phase TRUE height (for the fire-time pitch correction)
        if (last_target_ok) {
          double ang = std::atan2(yr - last_cy, xr - last_cx);
          // ang is center->plate (outward), while EKF x[6] is plate->center
          // (inward). Comparing them directly adds a 180deg offset, exactly
          // halfway between 120deg slots, so heights alias between plate IDs.
          const double base_outward = wrap_rad(last_a + CV_PI);
          int k = phase_bucket(ang - base_outward);
          zsum[k] += zr;
          zn[k]++;
        }
        // online 3-cluster of raw heights; z_mid = median cluster mean
        int ci = -1;
        for (int i = 0; i < zc_n; ++i)
          if (std::abs(zr - zc_mean[i]) < 0.06) { ci = i; break; }
        if (ci < 0 && zc_n < 3) { ci = zc_n++; zc_mean[ci] = zr; zc_cnt[ci] = 0; }
        if (ci < 0) {
          double best = 1e9;
          for (int i = 0; i < zc_n; ++i)
            if (std::abs(zr - zc_mean[i]) < best) { best = std::abs(zr - zc_mean[i]); ci = i; }
        }
        zc_cnt[ci]++;
        zc_mean[ci] += (zr - zc_mean[ci]) / std::min(zc_cnt[ci], 50);
        // substitution needs a settled center (>=1 drum period of samples)
        if (ctr_hist_x.size() < 50 || zc_n < 3) continue;
        double cx = 0, cy = 0;
        for (double v : ctr_hist_x) cx += v;
        for (double v : ctr_hist_y) cy += v;
        cx /= ctr_hist_x.size();
        cy /= ctr_hist_y.size();
        double rad = std::hypot(xr - cx, yr - cy);
        if (rad < 0.10 || rad > 0.45) continue;  // implausible radial -> pass through
        double a_ = zc_mean[0], b_ = zc_mean[1], c_ = zc_mean[2];
        double zmid = std::max(std::min(a_, b_), std::min(std::max(a_, b_), c_));
        double phi_raw = std::atan2(yr - cy, xr - cx);  // radial = plate normal azimuth
        double t_now_s = frame.t_present;
        // --- sign vote: independent least-squares slope for this height ---
        auto & samples = ep_samples[ci];
        // At the far perch one height may only be detected every few frames.
        // Up to 0.65 s is still unambiguous at 2.513 rad/s (< pi phase travel),
        // so retain the same-plate track across those detector gaps.
        const bool ep_break = ep_last_t[ci] < 0 || (t_now_s - ep_last_t[ci]) > 0.65;
        // Do not discard sparse observations here.  Besides the local slope
        // vote below, the known-|omega| hypothesis test needs long-baseline
        // samples and is invariant to whole-turn phase wrapping.
        double phi_unw = phi_raw;
        if (!samples.empty()) {  // unwrap only against the same physical plate
          double d = wrap_rad(phi_raw - ep_last_phi[ci]);
          phi_unw = samples.back().second + d;
        }
        samples.emplace_back(t_now_s, phi_unw);
        if (samples.size() > 80) samples.erase(samples.begin());
        ep_last_phi[ci] = phi_raw;
        ep_last_t[ci] = t_now_s;
        if (
          samples.size() >= 4 && samples.back().first - samples.front().first > 0.35 &&
          (ep_last_vote_t[ci] < 0 || t_now_s - ep_last_vote_t[ci] > 0.12)) {
          double tm = 0, pm = 0;
          for (const auto & s : samples) { tm += s.first; pm += s.second; }
          tm /= samples.size();
          pm /= samples.size();
          double num = 0, den = 0;
          for (const auto & s : samples) {
            num += (s.first - tm) * (s.second - pm);
            den += (s.first - tm) * (s.first - tm);
          }
          const double slope = den > 1e-9 ? num / den : 0;
          if (std::abs(slope) > 0.8 && std::abs(slope) < 5.0) {
            ph_votes.push_back(slope > 0 ? 1 : -1);
            if (ph_votes.size() > 9) ph_votes.pop_front();
            int vote_sum = 0;
            for (int v : ph_votes) vote_sum += v;
            // Local slopes are useful diagnostics but are too noisy to own the
            // lock.  Only the long-baseline known-speed hypothesis below may
            // select the per-round direction.
            ep_last_vote_t[ci] = t_now_s;
            tools::logger()->info(
              "[phase] cluster={} slope={:.2f} vote_sum={} sign={}", ci, slope, vote_sum,
              ph_sign);
          }
        }
        // Binary known-speed hypothesis.  For each height-identified plate,
        // de-rotate all observations with +/-|omega|.  Under the correct sign
        // they collapse to one circular phase; under the wrong sign they
        // spread around the circle.  Sum per-cluster resultant lengths because
        // each plate has its own unknown phase offset.
        if (ph_hyp_last_t < 0 || t_now_s - ph_hyp_last_t > 0.5) {
          ph_hyp_last_t = t_now_s;
          double score_pos_sum = 0, score_neg_sum = 0;
          int hyp_n = 0;
          double hyp_t_min = 1e100, hyp_t_max = -1e100;
          for (int hc = 0; hc < 3; ++hc) {
            const auto & hs = ep_samples[hc];
            if (hs.size() < 4) continue;
            const double t0 = hs.front().first;
            double pc = 0, ps = 0, nc = 0, ns = 0;
            for (const auto & obs : hs) {
              const double dt_h = obs.first - t0;
              const double rp = obs.second - kOmegaBook * dt_h;
              const double rn = obs.second + kOmegaBook * dt_h;
              pc += std::cos(rp);
              ps += std::sin(rp);
              nc += std::cos(rn);
              ns += std::sin(rn);
            }
            score_pos_sum += std::hypot(pc, ps);
            score_neg_sum += std::hypot(nc, ns);
            hyp_n += static_cast<int>(hs.size());
            hyp_t_min = std::min(hyp_t_min, hs.front().first);
            hyp_t_max = std::max(hyp_t_max, hs.back().first);
          }
          if (hyp_n >= 20) {
            const double score_pos = score_pos_sum / hyp_n;
            const double score_neg = score_neg_sum / hyp_n;
            const double margin = score_pos - score_neg;
            const double hyp_span = hyp_t_max - hyp_t_min;
            int vote_sum = 0;
            for (int v : ph_votes) vote_sum += v;
            const bool hypothesis_lock = std::abs(margin) > 0.10;
            const bool sustained_slope_lock = ph_votes.size() >= 7 && std::abs(vote_sum) >= 7;
            if (ph_sign == 0 && hyp_span >= 8.0 && (hypothesis_lock || sustained_slope_lock)) {
              ph_sign = hypothesis_lock ? (margin > 0 ? 1 : -1) : (vote_sum > 0 ? 1 : -1);
              std::fill(std::begin(zsum), std::end(zsum), 0.0);
              std::fill(std::begin(zn), std::end(zn), 0);
              ph_cx = cx;
              ph_cy = cy;
              ph_zmid = zmid;
              ph_geometry_frozen = true;
            }
            tools::logger()->info(
              "[phase-hyp] n={} span={:.1f} R+={:.3f} R-={:.3f} margin={:.3f} sign={}",
              hyp_n, hyp_span, score_pos, score_neg, margin, ph_sign);
          }
        }
        double phi_use = phi_raw;
        if (ph_sign != 0) {
          // predict-and-fold filter: advance at the rulebook rate, fold the
          // measurement onto the nearest 120° plate slot, correct softly.
          if (ph_t < 0) {
            ph_hat = phi_raw;
          } else {
            ph_hat += ph_sign * kOmegaBook * (t_now_s - ph_t);
            double innov = wrap_rad(phi_raw - ph_hat);
            innov = wrap_rad(std::fmod(innov + CV_PI / 3.0, 2.0 * CV_PI / 3.0) - CV_PI / 3.0);
            ph_hat = wrap_rad(ph_hat + 0.02 * innov);
          }
          ph_t = t_now_s;
          // this armor's own slot relative to the filtered phase
          double slot = std::round(wrap_rad(phi_raw - ph_hat) / (2.0 * CV_PI / 3.0));
          phi_use = wrap_rad(ph_hat + slot * 2.0 * CV_PI / 3.0);
        }
        // Fully model-consistent synthetic measurement: position snapped onto
        // the rulebook ring at the filtered phase (raw transverse noise 0.1-
        // 0.25m vs r=0.2765 otherwise fights the clean yaw inside the EKF and
        // keeps NIS above the 40% reset gate).
        const double center_x = ph_geometry_frozen ? ph_cx : cx;
        const double center_y = ph_geometry_frozen ? ph_cy : cy;
        const double center_z = ph_geometry_frozen ? ph_zmid : zmid;
        Eigen::Vector3d pos{xr, yr, center_z};
        if (ph_sign != 0) {
          pos[0] = center_x + 0.2765 * std::cos(phi_use);
          pos[1] = center_y + 0.2765 * std::sin(phi_use);
        }
        // Target::h_armor_xyz uses position = center - r*[cos(yaw),sin(yaw)],
        // so its yaw is the INWARD normal (plate -> center).  phi_use is the
        // outward ring phase (center -> plate); feeding it directly displaced
        // the initialized center by 2r and made every update structurally
        // inconsistent, producing the persistent high NIS/reset loop.
        const double yaw_inward = wrap_rad(phi_use + CV_PI);
        auto pts = solver.reproject_armor(pos, yaw_inward, armor.type, armor.name);
        if (pts.size() == 4) {
          for (int i = 0; i < 4; ++i) armor.points[i] = pts[i];
          armor.xyz_in_world = pos;
          armor.ypr_in_world = {yaw_inward, -15.0 * CV_PI / 180.0, 0};
          armor.ypd_in_world = tools::xyz2ypd(pos);
          armor.yaw_raw = yaw_inward;
          armor.pose_override = true;
        }
      }
      // Do not initialize/update the EKF with raw outpost poses while the
      // random spin direction is still unobservable.  Those poses are exactly
      // what previously seeded the filter with alternating +/-omega and high
      // NIS.  The frame in which the vote first locks is also discarded so
      // every measurement admitted to the tracker was synthesized under one
      // consistent phase model from the start of that frame.
      if (!phase_locked_at_frame_start) {
        armors.clear();
      } else {
        // Never mix raw PnP and constrained synthetic poses in one tracker.
        // A detection that failed center/radius/cluster validation above keeps
        // pose_override=false and must wait for a later frame.
        armors.remove_if([](const auto_aim::Armor & a) { return !a.pose_override; });
      }
    }
    auto targets = tracker.track(armors, t);

    double bullet_speed = link.attr(sentry, ga::kShooterRealSpeed).value_or(0) / 100.0;
    if (bullet_speed < 14 || bullet_speed > 30) bullet_speed = 24.5;

    auto command = aimer.aim(targets, t, bullet_speed);  // to_now=true, like standard.cpp
    // Reprojection is diagnostic telemetry only. The bridge must not override
    // the algorithm's command.control decision with a second state machine.
    double reproj_error_px = 1e9;
    double reproj_dx_px = 1e9, reproj_dy_px = 1e9;
    if (!targets.empty() && !armors_ui.empty()) {
      const auto model = targets.front().armor_xyza_list();
      for (const auto & m : model) {
        const auto pts = solver.reproject_armor(
          m.head(3), m[3], targets.front().armor_type, targets.front().name);
        if (pts.size() != 4) continue;
        cv::Point2f mc(0.f, 0.f);
        for (const auto & p : pts) mc += p;
        mc *= 0.25f;
        for (const auto & raw : armors_ui) {
          if (raw.name != auto_aim::ArmorName::outpost) continue;
          const double d = cv::norm(mc - raw.center);
          if (d < reproj_error_px) {
            reproj_error_px = d;
            reproj_dx_px = std::abs(static_cast<double>(mc.x - raw.center.x));
            reproj_dy_px = std::abs(static_cast<double>(mc.y - raw.center.y));
          }
        }
      }
    }
    if (now - t_ekf_log > 1s) {  // EKF health: settles "is it converging/spinning"
      t_ekf_log = now;
      if (!targets.empty()) {
        auto tf = targets.front();  // copy; convergened() mutates a cached flag
        const auto & xs = tf.ekf_x();
        double nis_fail = 0;
        auto it = tf.ekf().data.find("recent_nis_failures");
        if (it != tf.ekf().data.end()) nis_fail = it->second;
        const double center_bearing = std::atan2(xs[2], xs[0]) * kRad2Deg;
        const double center_distance = std::hypot(xs[0], xs[2]);
        const double aim_bearing = aimer.debug_aim_point.valid
                                     ? std::atan2(
                                         aimer.debug_aim_point.xyza[1],
                                         aimer.debug_aim_point.xyza[0]) *
                                         kRad2Deg
                                     : 999.0;
        const double aim_distance = aimer.debug_aim_point.valid
                                      ? std::hypot(
                                          aimer.debug_aim_point.xyza[0],
                                          aimer.debug_aim_point.xyza[1])
                                      : -1.0;
        double id0_width = -1.0, id0_height = -1.0;
        const auto model_armors = tf.armor_xyza_list();
        if (!model_armors.empty()) {
          const auto id0_pts = solver.reproject_armor(
            model_armors[0].head(3), model_armors[0][3], tf.armor_type, tf.name);
          if (id0_pts.size() == 4) {
            id0_width = 0.5 *
                        (cv::norm(id0_pts[0] - id0_pts[1]) +
                         cv::norm(id0_pts[3] - id0_pts[2]));
            id0_height = 0.5 *
                         (cv::norm(id0_pts[0] - id0_pts[3]) +
                          cv::norm(id0_pts[1] - id0_pts[2]));
          }
        }
        double aim_angle = 999.0, aim_width = -1.0, aim_height = -1.0;
        if (aimer.debug_aim_point.valid) {
          aim_angle = aimer.debug_aim_point.xyza[3] * kRad2Deg;
          const auto aim_pts = solver.reproject_armor(
            aimer.debug_aim_point.xyza.head(3), aimer.debug_aim_point.xyza[3],
            tf.armor_type, tf.name);
          if (aim_pts.size() == 4) {
            aim_width = 0.5 *
                        (cv::norm(aim_pts[0] - aim_pts[1]) +
                         cv::norm(aim_pts[3] - aim_pts[2]));
            aim_height = 0.5 *
                         (cv::norm(aim_pts[0] - aim_pts[3]) +
                          cv::norm(aim_pts[1] - aim_pts[2]));
          }
        }
        tools::logger()->info(
          "[ekf] a={:.1f} w={:.2f} r={:.3f} z=[{:.3f},{:.3f},{:.3f}] cd={:.2f} cb={:.1f} "
          "id0_wh={:.1f}x{:.1f} aim_d={:.2f} aim_b={:.1f} aim_a={:.1f} "
          "aim_wh={:.1f}x{:.1f} reproj_xy={:.1f}/{:.1f}px "
          "zmap=[{:.3f}/{},{:.3f}/{},{:.3f}/{}] state={} conv={} nis_fail={:.2f} "
          "det[r/b/e/p]={}/{}/{}/{}",
          xs[6] * kRad2Deg, xs[7], xs[8],
          model_armors.size() > 0 ? model_armors[0][2] : xs[4],
          model_armors.size() > 1 ? model_armors[1][2] : xs[4],
          model_armors.size() > 2 ? model_armors[2][2] : xs[4],
          center_distance, center_bearing, id0_width,
          id0_height, aim_distance, aim_bearing, aim_angle, aim_width, aim_height, reproj_dx_px,
          reproj_dy_px, zn[0] ? zsum[0] / zn[0] : 0.0, zn[0],
          zn[1] ? zsum[1] / zn[1] : 0.0, zn[1], zn[2] ? zsum[2] / zn[2] : 0.0, zn[2],
          tracker.state(), tf.convergened(), nis_fail, det_color_hist[0], det_color_hist[1],
          det_color_hist[2], det_color_hist[3]);
      } else {
        tools::logger()->info(
          "[ekf] no-target state={} det[r/b/e/p]={}/{}/{}/{}", tracker.state(),
          det_color_hist[0], det_color_hist[1], det_color_hist[2], det_color_hist[3]);
      }
    }

    if (!targets.empty()) {
      const auto & xs = targets.front().ekf_x();
      last_cx = xs[0];
      last_cy = xs[2];
      last_a = xs[6];
      last_target_ok = true;
    } else {
      last_target_ok = false;
    }

    // ---- OBSERVE mode: sweep bench — no aiming/fire commands at all ----
    if (mode_observe) {
      auto obs_now = std::chrono::steady_clock::now();
      if (sweep_idx < sweep.size()) {
        if (!sweep_phase_started) {
          const auto & p = sweep[sweep_idx];
          if (!p.camera.empty()) link.apply_camera(p.camera);
          if (!p.exec.empty()) {  // '|'-separated console commands
            std::string rest = p.exec;
            size_t bar;
            while ((bar = rest.find('|')) != std::string::npos) {
              link.exec(rest.substr(0, bar));
              rest = rest.substr(bar + 1);
            }
            if (!rest.empty()) link.exec(rest);
          }
          std::this_thread::sleep_for(2500ms);  // settle render/exposure
          sweep_t0 = std::chrono::steady_clock::now();
          sw_frames = sw_det = 0;
          sw_conf = sw_width = 0;
          sweep_phase_started = true;
          tools::logger()->info("[sweep] >>> {}", p.name);
        } else {
          sw_frames++;
          if (!armors_ui.empty()) {
            sw_det++;
            const auto & a = armors_ui.front();
            sw_conf += a.confidence;
            sw_width += (cv::norm(a.points[0] - a.points[1]) +
                         cv::norm(a.points[3] - a.points[2])) /
                        2.0;
            // Per-frame plate-yaw time series: is the PnP plate normal (the
            // EKF's rotation measurement) actually rotating? Settles whether
            // omega-unobservability is structural (solver tilt-model mismatch)
            // or pixel-starved. Uses the first solved armor.
            if (rec_fp && !armors.empty() && armors.front().xyz_in_world.norm() > 0.1) {
              const auto & m = armors.front();
              std::fprintf(
                rec_fp,
                "{\"t\":%.3f,\"phase\":\"%s\",\"ypr0\":%.4f,\"yaw_raw\":%.4f,"
                "\"bearing\":%.4f,\"dist\":%.3f,\"z\":%.3f,\"conf\":%.2f}\n",
                frame.t_present, sweep[sweep_idx].name.c_str(), m.ypr_in_world[0], m.yaw_raw,
                std::atan2(m.xyz_in_world[1], m.xyz_in_world[0]),
                std::hypot(m.xyz_in_world[0], m.xyz_in_world[1]), m.xyz_in_world[2],
                m.confidence);
            }
          }
          if (!targets.empty()) sw_last_w = targets.front().ekf_x()[7];
          sw_last_state = tracker.state();
          if (std::chrono::duration<double>(obs_now - sweep_t0).count() >= sweep[sweep_idx].secs) {
            tools::logger()->info(
              "[sweep] <<< {} det_rate={:.3f} conf={:.2f} width_px={:.1f} n={} "
              "ekf_w={:.2f} state={} img={}x{}",
              sweep[sweep_idx].name, sw_frames ? 1.0 * sw_det / sw_frames : 0,
              sw_det ? sw_conf / sw_det : 0, sw_det ? sw_width / sw_det : 0, sw_det, sw_last_w,
              sw_last_state, cap.width(), cap.height());
            if (rec_fp)
              std::fprintf(
                rec_fp,
                "{\"phase\":\"%s\",\"det_rate\":%.4f,\"conf\":%.3f,\"width_px\":%.1f,"
                "\"n\":%d,\"ekf_w\":%.3f,\"state\":\"%s\",\"w\":%d,\"h\":%d}\n",
                sweep[sweep_idx].name.c_str(), sw_frames ? 1.0 * sw_det / sw_frames : 0,
                sw_det ? sw_conf / sw_det : 0, sw_det ? sw_width / sw_det : 0, sw_det, sw_last_w,
                sw_last_state.c_str(), cap.width(), cap.height());
            std::fflush(rec_fp);
            sweep_idx++;
            sweep_phase_started = false;
          }
        }
      } else if (!sweep.empty()) {
        tools::logger()->info("[observe] sweep complete");
        break;
      }
      // debug window still updates below; skip all turret/fire command paths
    }

    bool fire = false;
    if (mode_observe) {
      // no commands in observe mode
    }
    if (!mode_observe && command.control) {
      Eigen::Vector3d gimbal_pos{tyaw, tpitch, 0};
      fire = allow_fire && shooter.shoot(command, aimer, targets, gimbal_pos);
      double ue_pitch_up_deg = -command.pitch * kRad2Deg;  // aimer is down-positive
      // z-adapter correction: the aimer's plate z is the EKF's single height;
      // replace with the learned height of the SPECIFIC plate it chose.
      if (z_adapter && !targets.empty() && aimer.debug_aim_point.valid) {
        const auto & xs = targets.front().ekf_x();
        const auto & ap = aimer.debug_aim_point.xyza;
        double horiz = std::hypot(ap[0], ap[1]);
        double lead_s = cfg_delay + horiz / bullet_speed;
        int ka = phase_bucket(ap[3] - (xs[6] + xs[7] * lead_s));
        bool omega_locked = std::abs(std::abs(xs[7]) - 2.513) < 0.13;
        if (omega_locked && zn[ka] >= 8 && horiz > 0.5) {
          double dz = zsum[ka] / zn[ka] - ap[2];
          ue_pitch_up_deg += std::atan2(dz, horiz) * kRad2Deg;
        }
      }
      const double cmd_yaw_deg_internal = command.yaw * kRad2Deg;
      const double cmd_yaw_deg = -cmd_yaw_deg_internal;  // RH sp_vision -> LH UE
      const double cmd_pitch_deg = ue_pitch_up_deg;
      link.exec(fmt::format(
        "UEExec RBExtAim {} {:.2f} {:.2f} {}", ext_pid, cmd_yaw_deg, cmd_pitch_deg,
        fire ? 1 : 0));
      last_fire_sent = fire;
      cmds++;
      if (fire) fire_cmds++;
      last_cmd = command;
      t_last_ctrl = now;
      scan_active = false;
    } else if (bench_match && !mode_observe) {
      // Release immediately on the first control->lost transition. Waiting for
      // the 2-second idle scan would otherwise leave a prior trigger pull held.
      release_fire();
      const double lost_s = std::chrono::duration<double>(now - t_last_ctrl).count();
      if (lost_s > kScanAfterLostSec) {
        if (!scan_active) {
          scan_yaw_deg = link.attr(sentry, ga::kTurretYaw).value_or(0.0);
          // Keep the real pitch even when it starts outside the scan band. The
          // first scan commands then walk it smoothly back towards +/-15 deg
          // instead of making a discontinuous clamp-sized jump.
          scan_pitch_deg = link.attr(sentry, ga::kTurretPitch).value_or(0.0);
          if (scan_pitch_deg > kScanPitchMaxDeg)
            scan_pitch_dir = -1;
          else if (scan_pitch_deg < -kScanPitchMaxDeg)
            scan_pitch_dir = 1;
          t_last_scan_step = now;
          scan_active = true;
        }
        const double scan_dt_s = std::clamp(
          std::chrono::duration<double>(now - t_last_scan_step).count(), 0.0,
          kScanMaxStepSec);
        // Advance the clock even while frozen. A detection that holds for a
        // while must not accumulate elapsed time and then cause a catch-up jump
        // on the first clear frame.
        t_last_scan_step = now;
        const bool raw_detection = !armors_ui.empty();
        if (!raw_detection) {
          scan_yaw_deg += kScanRateDegS * scan_dt_s;

          const bool pitch_was_outside =
            std::abs(scan_pitch_deg) > kScanPitchMaxDeg;
          if (scan_pitch_deg > kScanPitchMaxDeg)
            scan_pitch_dir = -1;
          else if (scan_pitch_deg < -kScanPitchMaxDeg)
            scan_pitch_dir = 1;
          scan_pitch_deg += scan_pitch_dir * kScanPitchRateDegS * scan_dt_s;

          // Once inside the band, run the normal triangle wave. While outside,
          // do not clamp: the inward ramp above is the smooth recovery path.
          if (!pitch_was_outside) {
            if (scan_pitch_deg >= kScanPitchMaxDeg) {
              scan_pitch_deg = kScanPitchMaxDeg;
              scan_pitch_dir = -1;
            } else if (scan_pitch_deg <= -kScanPitchMaxDeg) {
              scan_pitch_deg = -kScanPitchMaxDeg;
              scan_pitch_dir = 1;
            }
          }
        } else {
          // Freeze at the pose the turret has actually reached on this
          // detection frame. Reusing scan_yaw/pitch would keep commanding the
          // previous scan target (potentially up to one capped step ahead)
          // even though the velocity feed-forward was zeroed.
          const auto hold_yaw = link.attr(sentry, ga::kTurretYaw);
          const auto hold_pitch = link.attr(sentry, ga::kTurretPitch);
          if (hold_yaw && hold_pitch) {
            scan_yaw_deg = *hold_yaw;
            scan_pitch_deg = *hold_pitch;
          } else {
            // With a raw visual contact but no current pose telemetry, do not
            // issue a stale absolute target. The native fire watchdog and the
            // release above leave this as the safe hold path.
            scan_active = false;
          }
        }
        if (scan_active) {
          scan_yaw_deg = std::remainder(scan_yaw_deg, 360.0);
          link.exec(fmt::format(
            "UEExec RBExtAim {} {:.1f} {:.1f} 0 {:.0f} {:.0f}", ext_pid, scan_yaw_deg,
            scan_pitch_deg, raw_detection ? 0.0 : kScanRateDegS,
            raw_detection ? 0.0 : scan_pitch_dir * kScanPitchRateDegS));
          last_fire_sent = false;
          cmds++;
        }
      }
    } else {
      // Non-match benches do not have an idle scan, but still need the same
      // immediate release when the aimer drops control or observe mode is used.
      release_fire();
    }

    // ---------- sp_vision's own debug window (auto_aim_test.cpp style) ----------
    if (ui_quit) break;
    cv::Mat img = frame.img;  // annotate in place; frame is a deep copy already
    // Raw yolo verdicts, same labels as the original "detection" window
    // (yolo11.cpp draw_detections: conf color name type) — drawn from the
    // pre-track snapshot so filtered/misclassified plates stay visible.
    for (const auto & a : armors_ui) {
      std::vector<cv::Point2f> pts(a.points.begin(), a.points.end());
      tools::draw_points(img, pts, {255, 200, 0});
      tools::draw_text(
        img,
        fmt::format("{:.2f} {} {} {}", a.confidence, auto_aim::COLORS[a.color],
                    auto_aim::ARMOR_NAMES[a.name], auto_aim::ARMOR_TYPES[a.type]),
        {static_cast<int>(a.center.x) - 40, static_cast<int>(a.center.y) - 14},
        {255, 200, 0}, 0.5);
    }
    // Own status: health, revive accounting and running combat effectiveness.
    tools::draw_text(
      img,
      fmt::format("self hp:{:.0f} lives_lost:{} dealt:{:.0f} last_revive_wait:{:.0f}s",
                  bench_match ? hp : link.attr(sentry, ga::kHealth).value_or(-1), lives_lost,
                  dealt_carry + std::max(0.0, last_dealt - dealt_base), last_revive_wait_s),
      {10, 30}, {0, 200, 255});
    tools::draw_text(
      img,
      fmt::format(
        "command {} yaw{:.2f} pitch{:.2f} shoot:{}", command.control, -command.yaw * 57.3,
        command.pitch * 57.3, fire),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img, fmt::format("turret yaw{:.2f} pitch{:.2f} v{:.1f}", tyaw_ue * 57.3, tpitch * 57.3,
                       bullet_speed),
      {10, 90}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format("state:{} det:{}/{} hp:{:.0f} allow:{:.0f} shots:{:.0f}",
                  tracker.state(), det_frames, frames, hp, allow, fired),
      {10, 120}, {0, 255, 255});

    if (!targets.empty()) {
      auto target = targets.front();
      const auto & tx = target.ekf_x();
      for (Eigen::Vector4d xyza : target.armor_xyza_list()) {
        const int k = phase_bucket(xyza[3] - tx[6]);
        if (zn[k] >= 8) xyza[2] = zsum[k] / zn[k];
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }
      auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        Eigen::Vector4d draw_aim = aim_point.xyza;
        double horiz = std::hypot(draw_aim[0], draw_aim[1]);
        double lead_s = cfg_delay + horiz / bullet_speed;
        int k = phase_bucket(draw_aim[3] - (tx[6] + tx[7] * lead_s));
        if (zn[k] >= 8) draw_aim[2] = zsum[k] / zn[k];
        auto image_points = solver.reproject_armor(
          draw_aim.head(3), draw_aim[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
      }
      const auto & x = target.ekf_x();
      tools::draw_text(
        img, fmt::format("ekf w{:.2f} r{:.3f} d{:.2f}", x[7], x[8],
                         std::hypot(x[0], x[2])),
        {10, 150}, {0, 255, 0});
    }

    {
      std::lock_guard<std::mutex> lk(ui_mu);
      img.copyTo(ui_frame);  // hand off to the UI thread (latest-wins mailbox)
    }
    if (dump_ui_every > 0 && frames % dump_ui_every == 0)
      cv::imwrite(fmt::format("ui_dump/ui_{:06d}.jpg", frames), img);

    if (now - t_log > 5s) {
      t_log = now;
      auto t_frame_end = std::chrono::steady_clock::now();
      tools::logger()->info(
        "[gestalt] frames={} det={} cmds={} fire_cmds={} shots={:.0f} hp={:.0f} allow={:.0f} "
        "det_ms={:.0f} post_ms={:.0f} grab_avg={:.0f} rej_avg={:.0f} rej_n={}",
        frames, det_frames, cmds, fire_cmds, fired, hp, allow,
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count(),
        std::chrono::duration<double, std::milli>(t_frame_end - t_det1).count(),
        grab_n ? grab_acc / grab_n : 0, rej_n ? rej_acc / rej_n : 0, rej_n);
      grab_acc = rej_acc = 0;
      grab_n = rej_n = 0;
    }
  }

  // Latch natural settlement before teardown: MatchStatus may return to prep
  // after its short settled-state display window.
  observed_match_end = observed_match_end || (bench_match && link.match_status() >= 2);
  if (bench_match && observed_match_end && current_life_accounted) {
    terminal_inactive_release_allowed = true;
  }

  // Teardown is a hard safety boundary: clear the latched trigger before the
  // UI join, final telemetry delay, and ExtAimClaim release.
  release_fire(true);
  ui_quit = true;
  if (ui_thread.joinable()) ui_thread.join();

  // Let in-flight rounds land before the final HP read (≤6m ⇒ <0.5s fly time).
  std::this_thread::sleep_for(1500ms);
  const double hp_final = probe > 0 ? link.attr(probe, ga::kHealth).value_or(0) : -1;
  const double allow_final = link.attr(sentry, ga::kAllowance17mm).value_or(0);
  const auto bf_final = link.attr(sentry, ga::kBulletsFired);
  const double used =
    fired_carry +
    (current_life_accounted
       ? 0.0
       : (bf_final ? std::max(0.0, *bf_final - fired0)
                   : std::max(0.0, allowance_base - allow_final)));
  if (const auto dd = link.attr(sentry, ga::kDamageApplied)) last_dealt = *dd;
  const double dealt =
    dealt_carry +
    (current_life_accounted ? 0.0 : std::max(0.0, last_dealt - dealt_base));
  // Bench mode infers hits from target HP loss. Match mode has no single
  // probe target, so use the game's cumulative damage telemetry instead.
  const double hits = bench_match
    ? dealt / bench_damage_per_hit
    : (hp0 - hp_final) / bench_damage_per_hit;
  const auto final_pid_map = link.find_player(0);
  const bool final_identity_ok =
    identity_contract_ok &&
    (terminal_inactive_release_allowed ||
     (final_pid_map && is_expected_match_sentry(*final_pid_map)));
  if (bench_match && !final_identity_ok) {
    tools::logger()->error(
      "[gestalt] RESULT identity gate failed: pid0_map={} controlled_map={} class={} team={} "
      "config={}",
      final_pid_map.value_or(-1), sentry,
      final_pid_map ? link.attr(*final_pid_map, ga::kClass).value_or(-1) : -1,
      final_pid_map ? link.attr(*final_pid_map, ga::kTeamId).value_or(-1) : -1,
      link.player_entity_config(0).value_or(-1));
  }
  const bool release_restored = release_takeover("teardown");
  if (bench_vehicle) {
    // Teardown: never kill the target — a corpse keeps extinguished-but-
    // detectable plates at the test point and poisons the next round's EKF
    // (observed: w collapsed to 0.02 and 100 rounds hit the corpse for zero
    // damage). A live car left standing can still body-block later target
    // drives, so prefer parking it at a configured out-of-the-way spot.
    if (bench_target_park_cm.size() >= 2) {
      // Stop the lab's per-tick mode maintenance FIRST — an active mode-99
      // spin re-asserts itself every tick and overrides the park goto
      // (observed: the hero never left the test point).
      link.exec(fmt::format("RBNavLab mode {} 0 0", bench_target_pid));
      link.exec(fmt::format("RBNavLab goto {} {} {}", bench_target_pid,
                            static_cast<int>(bench_target_park_cm[0]),
                            static_cast<int>(bench_target_park_cm[1])));
    } else {
      link.exec(fmt::format("RBNavLab mode {} 0 0", bench_target_pid));
      link.exec(fmt::format("RBNavLab clear {}", bench_target_pid));
    }
  }
  if (!hp_drop_hist.empty()) {
    std::string drops;
    for (const auto & [delta, n] : hp_drop_hist) drops += fmt::format("{}x-{} ", n, delta);
    tools::logger()->info(
      "[gestalt] hp drop histogram (calibrates bench_damage_per_hit={}): {}",
      bench_damage_per_hit, drops);
  }
  const int final_match_status = link.match_status();
  const bool ws_continuity_gate =
    ws_continuity_clean && link.connected() &&
    link.connection_generation() == ws_generation;
  // A single 50ms miss is normal scheduling jitter. Ten consecutive misses is
  // the declared interruption threshold that requires a full re-arm.
  const bool capture_recovery_gate = !capture_recovery_pending && grab_fail < 10;
  const bool match_pass = bench_match && observed_match_end && frames > 0 &&
                           det_frames > 0 && used > 0 && dealt > 0 &&
                           takeover_contract_ok && identity_contract_ok && final_identity_ok &&
                           frame_writer_contract_ok && frame_identity_contract_ok &&
                           ws_continuity_gate && capture_recovery_gate && release_restored;
  const char * result = bench_match
    ? (match_pass ? "MATCH_COMPLETE" : "MATCH_FAILED")
    : (hp_final == 0 ? "DESTROYED" : "PARTIAL");
  tools::logger()->info(
    "[gestalt] RESULT={} hp {}->{} damaging_hits={} bullets={} hit_rate={:.3f} "
    "frames={} det_rate={:.3f} fire_cmds={} dmg_per_hit={} damage_dealt={:.0f} "
    "lives_lost={} match_status={} match_end_seen={} takeover_gate={} pid0_hachisen={} "
    "frame_writer_pid={} frame_writer_gate={} view_identity_gate={} takeover_epoch={} "
    "ws_continuity_gate={} capture_recovery_gate={} release_restored={}",
    result, hp0, hp_final, hits, used,
    used > 0 ? hits / used : 0.0, frames, frames > 0 ? 1.0 * det_frames / frames : 0.0,
    fire_cmds, bench_damage_per_hit, dealt, lives_lost, final_match_status, observed_match_end,
    takeover_contract_ok, final_identity_ok, cap.process_id(), frame_writer_contract_ok,
    frame_identity_contract_ok, verified_takeover_epoch, ws_continuity_gate,
    capture_recovery_gate, release_restored);
  return bench_match ? (match_pass ? 0 : 2) : (hp_final == 0 ? 0 : 2);
}
