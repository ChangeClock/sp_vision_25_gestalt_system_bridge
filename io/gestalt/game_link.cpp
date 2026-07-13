#include "game_link.hpp"

#include <easywsclient.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>

using easywsclient::WebSocket;
using json = nlohmann::json;

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
namespace
{
struct WinsockInit
{
  WinsockInit()
  {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
  }
  ~WinsockInit() { WSACleanup(); }
};
}  // namespace
#endif

namespace io::gestalt
{
GameLink::GameLink(int port) : port_(port)
{
#ifdef _WIN32
  static WinsockInit winsock_init;  // once per process, before any socket use
#endif
  reader_ = std::thread([this] { reader_loop(); });
}

GameLink::~GameLink()
{
  stop_ = true;
  if (reader_.joinable()) reader_.join();
  std::lock_guard<std::mutex> lk(mu_);
  if (ws_) {
    static_cast<WebSocket *>(ws_)->close();
    delete static_cast<WebSocket *>(ws_);
    ws_ = nullptr;
  }
}

bool GameLink::connect_once()
{
  char url[64];
  std::snprintf(url, sizeof(url), "ws://127.0.0.1:%d/", port_);
  WebSocket * ws = WebSocket::from_url(url);
  if (!ws) return false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (ws_) delete static_cast<WebSocket *>(ws_);
    ws_ = ws;
    watched_.clear();
  }
  // Initial sweep: maps 1..256; ptr attrs then auto-discover vehicle maps.
  std::vector<int> initial;
  for (int i = 1; i <= 256; ++i) initial.push_back(i);
  watch(initial);
  connected_ = true;
  std::printf("[GameLink] connected %s\n", url);
  return true;
}

void GameLink::reader_loop()
{
  while (!stop_) {
    if (!connected_) {
      if (!connect_once()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
    }
    WebSocket * ws;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ws = static_cast<WebSocket *>(ws_);
    }
    if (!ws || ws->getReadyState() == WebSocket::CLOSED) {
      connected_ = false;
      continue;
    }
    {
      // easywsclient is not thread-safe: poll() flushes the tx buffer that
      // send() (via send_raw, same mutex) appends to — keep them serialized.
      // poll(1), NOT poll(5): the vision loop reads ~8 attrs per frame under
      // this same mutex; a 5ms hold starves it into ~130ms frame periods.
      std::lock_guard<std::mutex> lk(mu_);
      ws->poll(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // yield to attr readers
    std::vector<int> discovered;
    ws->dispatch([&](const std::string & raw) {
      json p = json::parse(raw, nullptr, /*allow_exceptions=*/false);
      if (p.is_discarded() || p.value("type", -1) != 0) return;
      if (p.value("method", "") != "watchAttributeMaps.result") return;
      auto results = p["params"]["watch_attribute_maps_results"];
      if (!results.is_array()) return;
      std::lock_guard<std::mutex> lk(mu_);
      const int pose_map = pose_map_.load();
      for (auto & r : results) {
        int mid = r.value("attribute_map_id", -1);
        if (mid < 0) continue;
        auto & m = maps_[mid];
        auto attrs = r["attributes"];
        if (!attrs.is_object()) continue;
        bool pose_touched = false;
        for (auto it = attrs.begin(); it != attrs.end(); ++it) {
          if (!it.value().is_number()) continue;
          int aid = std::stoi(it.key());
          double v = it.value().get<double>();
          m[aid] = v;
          if (aid == attr::kMapPtr && v > 0) discovered.push_back(static_cast<int>(v));
          if (mid == pose_map && (aid == attr::kTurretYaw || aid == attr::kTurretPitch))
            pose_touched = true;
        }
        if (pose_touched) {
          auto y = m.find(attr::kTurretYaw);
          auto p = m.find(attr::kTurretPitch);
          if (y != m.end() && p != m.end()) {
            double now_s =
              std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
            pose_hist_.push_back({now_s, y->second, p->second});
            if (pose_hist_.size() > 64) pose_hist_.pop_front();
          }
        }
      }
    });
    if (!discovered.empty()) watch(discovered);
  }
}

void GameLink::watch(const std::vector<int> & ids)
{
  json fresh = json::array();
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (int id : ids)
      if (watched_.insert(id).second) fresh.push_back(id);
  }
  if (fresh.empty()) return;
  json msg = {{"type", 0},
              {"id", ++next_id_},
              {"method", "attribute.watchAttributeMaps"},
              {"params", {{"attribute_map_ids", fresh}, {"watch_type", 2}}}};
  send_raw(msg.dump());
}

void GameLink::send_raw(const std::string & payload)
{
  std::lock_guard<std::mutex> lk(mu_);
  auto * ws = static_cast<WebSocket *>(ws_);
  if (ws && ws->getReadyState() == WebSocket::OPEN) ws->send(payload);
}

void GameLink::exec(const std::string & command)
{
  json msg = {{"type", 0},
              {"id", ++next_id_},
              {"method", "console.exec"},
              {"params", {{"command", command}}}};
  send_raw(msg.dump());
}

void GameLink::apply_camera(const std::string & camera_json)
{
  json cam = json::parse(camera_json, nullptr, false);
  if (cam.is_discarded()) return;
  json msg = {{"type", 0},
              {"id", ++next_id_},
              {"method", "rgbCamera.applySettings"},
              {"params", {{"camera", cam}}}};
  send_raw(msg.dump());
}

std::optional<double> GameLink::attr(int map_id, int attr_id) const
{
  std::lock_guard<std::mutex> lk(mu_);
  auto mit = maps_.find(map_id);
  if (mit == maps_.end()) return std::nullopt;
  auto ait = mit->second.find(attr_id);
  if (ait == mit->second.end()) return std::nullopt;
  return ait->second;
}

void GameLink::enable_pose_history(int map_id)
{
  std::lock_guard<std::mutex> lk(mu_);
  pose_map_ = map_id;
  pose_hist_.clear();
}

std::optional<std::pair<double, double>> GameLink::pose_at(double t_s) const
{
  std::lock_guard<std::mutex> lk(mu_);
  if (pose_hist_.empty()) return std::nullopt;
  if (pose_hist_.size() == 1 || t_s <= pose_hist_.front().t)
    return std::make_pair(pose_hist_.front().yaw, pose_hist_.front().pitch);
  auto lerp_yaw = [](double a, double b, double f) {
    double d = b - a;
    while (d > 180.0) d -= 360.0;
    while (d < -180.0) d += 360.0;
    return a + d * f;
  };
  for (size_t i = 1; i < pose_hist_.size(); ++i) {
    if (t_s <= pose_hist_[i].t) {
      const auto & a = pose_hist_[i - 1];
      const auto & b = pose_hist_[i];
      double f = (b.t - a.t) > 1e-6 ? (t_s - a.t) / (b.t - a.t) : 1.0;
      return std::make_pair(lerp_yaw(a.yaw, b.yaw, f), a.pitch + (b.pitch - a.pitch) * f);
    }
  }
  // Not bracketed (frame newer than the last push): extrapolate from the last
  // two samples, clamped to 60ms — beyond that just hold the last value.
  const auto & a = pose_hist_[pose_hist_.size() - 2];
  const auto & b = pose_hist_.back();
  double dt = std::min(t_s - b.t, 0.06);
  double span = b.t - a.t;
  if (span < 1e-6 || dt <= 0) return std::make_pair(b.yaw, b.pitch);
  double f = 1.0 + dt / span;
  return std::make_pair(lerp_yaw(a.yaw, b.yaw, f), a.pitch + (b.pitch - a.pitch) * f);
}

std::optional<int> GameLink::find_vehicle(int cls, int team) const
{
  std::lock_guard<std::mutex> lk(mu_);
  // Prefer the NEWEST alive body (highest map id): repeated Respawns leave
  // corpse maps with frozen alive-looking snapshots at LOWER ids — reading
  // those is the "dead sentry frozen telemetry" trap.
  std::optional<int> best_alive, fallback;
  for (auto & [mid, m] : maps_) {
    auto c = m.find(attr::kClass);
    auto t = m.find(attr::kTeamId);
    if (c == m.end() || t == m.end()) continue;
    if (static_cast<int>(c->second) != cls || static_cast<int>(t->second) != team) continue;
    auto hp = m.find(attr::kHealth);
    if (hp != m.end() && hp->second > 0)
      best_alive = mid;  // maps_ is id-ordered; keep overwriting → highest id
    else
      fallback = mid;
  }
  return best_alive ? best_alive : fallback;
}

std::optional<int> GameLink::find_player(int player_id) const
{
  std::lock_guard<std::mutex> lk(mu_);
  std::optional<int> best_alive, fallback;
  for (const auto & [mid, m] : maps_) {
    const auto p = m.find(attr::kPlayerId);
    if (p == m.end() || static_cast<int>(p->second) != player_id) continue;
    const auto hp = m.find(attr::kHealth);
    if (hp != m.end() && hp->second > 0)
      best_alive = mid;
    else
      fallback = mid;
  }
  return best_alive ? best_alive : fallback;
}

std::optional<int> GameLink::find_red_outpost() const
{
  std::lock_guard<std::mutex> lk(mu_);
  for (auto & [mid, m] : maps_) {
    auto g = m.find(attr::kOutpostRedGlobal);
    if (g != m.end() && g->second > 0) return static_cast<int>(g->second);
  }
  for (auto & [mid, m] : maps_) {
    auto hm = m.find(attr::kHealthMax);
    if (hm != m.end() && hm->second == 1500 && m.find(attr::kPlayerId) == m.end()) return mid;
  }
  return std::nullopt;
}

double GameLink::match_status() const
{
  std::lock_guard<std::mutex> lk(mu_);
  for (auto & [mid, m] : maps_) {
    auto it = m.find(attr::kMatchStatus);
    if (it != m.end()) return it->second;
  }
  return -1;
}

}  // namespace io::gestalt
