#include "shared_frame_capture.hpp"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace io::gestalt
{
namespace
{
constexpr uint64_t kMagic = 0x314D52464E565347ull;
constexpr uint32_t kVersion = 1;
constexpr uint32_t kRegionHeaderBytes = 4096;
constexpr uint32_t kSlotHeaderBytes = 256;

enum class PixelFormat : uint32_t
{
  bgra8 = 1,
  rgba8 = 2,
  a2b10g10r10 = 3,
};

struct alignas(64) RegionHeader
{
  uint64_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint32_t slot_count;
  uint32_t slot_header_bytes;
  uint32_t slot_stride;
  uint32_t max_width;
  uint32_t max_height;
  uint64_t region_bytes;
  uint64_t writer_process_id;
  uint64_t qpc_frequency;
  uint8_t reserved[kRegionHeaderBytes - 64];
};
static_assert(sizeof(RegionHeader) == kRegionHeaderBytes);

struct alignas(64) SlotHeader
{
  volatile int64_t commit_sequence;
  uint64_t engine_frame_counter;
  int64_t capture_qpc;
  double world_time_seconds;
  float delta_world_time_seconds;
  float camera_position_cm[3];
  float camera_quaternion_xyzw[4];
  float horizontal_fov_degrees;
  uint32_t view_x;
  uint32_t view_y;
  uint32_t width;
  uint32_t height;
  uint32_t row_bytes;
  uint32_t pixel_bytes;
  uint32_t pixel_format;
  uint32_t reserved0;
  uint8_t reserved[kSlotHeaderBytes - 112];
};
static_assert(sizeof(SlotHeader) == kSlotHeaderBytes);

struct FoundWindow
{
  HWND hwnd = nullptr;
  DWORD pid = 0;
  std::string title;
  bool has_frame_mapping = false;
};

std::wstring mapping_name(DWORD pid);

bool frame_mapping_exists(DWORD pid)
{
  const auto name = mapping_name(pid);
  HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
  if (!mapping) return false;
  CloseHandle(mapping);
  return true;
}

std::string exe_of_pid(DWORD pid)
{
  HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!process) return {};
  char path[MAX_PATH]{};
  DWORD size = MAX_PATH;
  std::string executable;
  if (QueryFullProcessImageNameA(process, 0, path, &size)) {
    executable.assign(path, size);
    auto slash = executable.find_last_of("\\/");
    if (slash != std::string::npos) executable = executable.substr(slash + 1);
    std::transform(executable.begin(), executable.end(), executable.begin(), ::tolower);
  }
  CloseHandle(process);
  return executable;
}

BOOL CALLBACK enum_window(HWND hwnd, LPARAM parameter)
{
  auto * found = reinterpret_cast<FoundWindow *>(parameter);
  if (!IsWindowVisible(hwnd)) return TRUE;
  char title[512]{};
  if (!GetWindowTextA(hwnd, title, sizeof(title)) || !title[0]) return TRUE;
  DWORD pid = 0;
  GetWindowThreadProcessId(hwnd, &pid);
  // The bridge executable is also named gestalt.exe. A visible console/debug
  // window owned by this process must never be mistaken for the game viewport.
  if (pid == GetCurrentProcessId()) return TRUE;
  const auto executable = exe_of_pid(pid);
  if (executable.rfind("unrealeditor", 0) != 0 && executable.rfind("gestalt", 0) != 0 &&
      executable.rfind("robotbridge", 0) != 0)
    return TRUE;
  // Multiple UE projects may be open on a development machine.  Process name
  // and window visibility are not sufficient to identify the game that owns
  // this bridge; prefer the process that actually publishes a VisionBridge
  // shared-memory ring.  Keep the first eligible window only as a startup
  // fallback while its mapping is being created.
  const bool has_frame_mapping = frame_mapping_exists(pid);
  if (found->hwnd && !has_frame_mapping) return TRUE;
  // A minimized UE viewport legitimately reports a 0x0 client rect. Process
  // identity is authoritative here; init/raise_window restores it before the
  // first frame is consumed.
  found->hwnd = hwnd;
  found->pid = pid;
  found->title = title;
  found->has_frame_mapping = has_frame_mapping;
  // A publisher is authoritative; do not let another unrelated UnrealEditor
  // window encountered later overwrite it.
  return has_frame_mapping ? FALSE : TRUE;
}

FoundWindow find_game_window()
{
  FoundWindow found;
  EnumWindows(enum_window, reinterpret_cast<LPARAM>(&found));
  return found;
}

std::wstring mapping_name(DWORD pid)
{
  wchar_t name[64]{};
  swprintf_s(name, L"{47534652-414D-4501-0000-%012X}", pid);
  return name;
}

double qpc_seconds(int64_t ticks, uint64_t frequency)
{
  return frequency ? static_cast<double>(ticks) / static_cast<double>(frequency) : 0.0;
}
}  // namespace

struct SharedFrameCapture::Impl
{
  HWND hwnd = nullptr;
  HANDLE mapping = nullptr;
  const uint8_t * base = nullptr;
  const RegionHeader * region = nullptr;

  void close()
  {
    if (base) UnmapViewOfFile(base);
    if (mapping) CloseHandle(mapping);
    base = nullptr;
    mapping = nullptr;
    region = nullptr;
  }
};

SharedFrameCapture::SharedFrameCapture() : impl_(new Impl) {}

SharedFrameCapture::~SharedFrameCapture()
{
  impl_->close();
  delete impl_;
}

bool SharedFrameCapture::init(int timeout_ms)
{
  impl_->close();
  const auto found = find_game_window();
  if (!found.hwnd) {
    std::fprintf(stderr, "[SharedFrameCapture] no eligible game viewport found\n");
    return false;
  }
  impl_->hwnd = found.hwnd;
  title_ = found.title;
  process_id_ = found.pid;
  ShowWindow(found.hwnd, SW_RESTORE);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  const auto name = mapping_name(found.pid);
  while (std::chrono::steady_clock::now() < deadline) {
    impl_->mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
    if (impl_->mapping) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  if (!impl_->mapping) {
    std::fprintf(
      stderr, "[SharedFrameCapture] mapping unavailable pid=%lu title='%s'\n",
      static_cast<unsigned long>(found.pid), found.title.c_str());
    return false;
  }

  impl_->base = static_cast<const uint8_t *>(MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0));
  if (!impl_->base) {
    impl_->close();
    return false;
  }
  impl_->region = reinterpret_cast<const RegionHeader *>(impl_->base);
  if (impl_->region->magic != kMagic || impl_->region->version != kVersion ||
      impl_->region->header_bytes != kRegionHeaderBytes ||
      impl_->region->slot_header_bytes != kSlotHeaderBytes || impl_->region->slot_count == 0 ||
      impl_->region->slot_count > 16 || impl_->region->writer_process_id != found.pid) {
    std::fprintf(
      stderr, "[SharedFrameCapture] invalid mapping header pid=%lu title='%s'\n",
      static_cast<unsigned long>(found.pid), found.title.c_str());
    impl_->close();
    return false;
  }
  last_commit_ = 0;
  return true;
}

void SharedFrameCapture::raise_window()
{
  if (!impl_->hwnd) return;
  ShowWindow(impl_->hwnd, SW_RESTORE);
  SetWindowPos(
    impl_->hwnd, HWND_TOPMOST, 0, 0, 0, 0,
    SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
  SetForegroundWindow(impl_->hwnd);
}

CapturedFrame SharedFrameCapture::grab(int timeout_ms)
{
  CapturedFrame out;
  if (!impl_->region) return out;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  while (std::chrono::steady_clock::now() < deadline) {
    const SlotHeader * best = nullptr;
    uint64_t best_commit = last_commit_;
    const uint8_t * best_pixels = nullptr;
    for (uint32_t i = 0; i < impl_->region->slot_count; ++i) {
      const uint8_t * slot_base = impl_->base + impl_->region->header_bytes +
                                  static_cast<size_t>(i) * impl_->region->slot_stride;
      const auto * slot = reinterpret_cast<const SlotHeader *>(slot_base);
      // Aligned 64-bit loads are atomic on x64. The mapping is deliberately
      // FILE_MAP_READ, so an InterlockedCompareExchange (an RMW instruction)
      // would fault even when exchange/comparand are both zero.
      const int64_t commit = slot->commit_sequence;
      MemoryBarrier();
      if (commit <= 0 || (commit & 1) != 0 || static_cast<uint64_t>(commit) <= best_commit) continue;
      if (slot->width == 0 || slot->height == 0 || slot->width > impl_->region->max_width ||
          slot->height > impl_->region->max_height ||
          slot->row_bytes != slot->width * 4 ||
          slot->pixel_bytes != slot->row_bytes * slot->height ||
          slot->pixel_bytes > impl_->region->slot_stride - impl_->region->slot_header_bytes)
        continue;
      best = slot;
      best_commit = static_cast<uint64_t>(commit);
      best_pixels = slot_base + impl_->region->slot_header_bytes;
    }

    if (!best) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    SlotHeader metadata = *best;
    std::vector<uint8_t> pixels(metadata.pixel_bytes);
    std::memcpy(pixels.data(), best_pixels, pixels.size());
    MemoryBarrier();
    const int64_t commit_after = best->commit_sequence;
    if (commit_after != static_cast<int64_t>(best_commit) || (commit_after & 1) != 0) continue;

    cv::Mat source(
      static_cast<int>(metadata.height), static_cast<int>(metadata.width), CV_8UC4, pixels.data(),
      metadata.row_bytes);
    if (metadata.pixel_format == static_cast<uint32_t>(PixelFormat::bgra8))
      cv::cvtColor(source, out.img, cv::COLOR_BGRA2BGR);
    else if (metadata.pixel_format == static_cast<uint32_t>(PixelFormat::rgba8))
      cv::cvtColor(source, out.img, cv::COLOR_RGBA2BGR);
    else if (metadata.pixel_format == static_cast<uint32_t>(PixelFormat::a2b10g10r10)) {
      out.img.create(static_cast<int>(metadata.height), static_cast<int>(metadata.width), CV_8UC3);
      for (uint32_t y = 0; y < metadata.height; ++y) {
        const auto * packed = reinterpret_cast<const uint32_t *>(
          pixels.data() + static_cast<size_t>(y) * metadata.row_bytes);
        auto * bgr = out.img.ptr<cv::Vec3b>(static_cast<int>(y));
        for (uint32_t x = 0; x < metadata.width; ++x) {
          const uint32_t value = packed[x];
          bgr[x][2] = static_cast<uint8_t>(((value >> 0) & 0x3ffu) * 255u / 1023u);
          bgr[x][1] = static_cast<uint8_t>(((value >> 10) & 0x3ffu) * 255u / 1023u);
          bgr[x][0] = static_cast<uint8_t>(((value >> 20) & 0x3ffu) * 255u / 1023u);
        }
      }
    }
    else
      return out;

    width_ = static_cast<int>(metadata.width);
    height_ = static_cast<int>(metadata.height);
    out.t_present = qpc_seconds(metadata.capture_qpc, impl_->region->qpc_frequency);
    out.frame_id = metadata.engine_frame_counter;
    for (int i = 0; i < 3; ++i) out.camera_position_ue_cm[i] = metadata.camera_position_cm[i];
    for (int i = 0; i < 4; ++i)
      out.camera_quaternion_ue_xyzw[i] = metadata.camera_quaternion_xyzw[i];
    out.fov_degrees = metadata.horizontal_fov_degrees;
    out.has_camera_pose = true;
    out.ok = true;
    last_commit_ = best_commit;
    return out;
  }
  return out;
}
}  // namespace io::gestalt
