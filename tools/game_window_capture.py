#!/usr/bin/env python3
"""Game-window frame recorder for the gestalt_system bridge (V1 data collection,
later the live camera source for GameIO).

Captures the UE game window via Desktop Duplication (dxcam) — zero in-engine
cost — while a WebSocket thread tracks the game's telemetry (turret yaw/pitch,
match status) and drives setup over console.exec:
  - Respawn pid0 as a blue-sentry body (FIRST-PERSON gun view, no TPS force)
  - hand the body to the built-in AI (IsAIControlled=1 + mode words keeper)
  - apply realistic camera params per phase via rgbCamera.applySettings
    (Hik MV-CS016 + 6mm lens ~= 45deg horizontal FOV, low exposure so the
    armor lightbars dominate like a 2ms industrial-camera exposure)

Frames land in <out>/<phase>/f%06d.jpg with a per-phase JSONL sidecar
{idx, t, yaw, pitch}. Launch the match with ai-match-selftest.ps1 -HudHidden 1
so captures contain pure gameplay pixels.

Usage: python tools/game_window_capture.py <wsPort> [--out DIR] [--fps 15]
"""

import argparse
import ctypes
import json
import os
import threading
import time

import cv2
import dxcam
import win32gui
import win32process
from websockets.sync.client import connect as ws_connect

A_PLAYER_ID = 10000035
A_TEAM_ID = 10000036
A_CLASS = 60000002
A_POS_X = 10000107
A_POS_Y = 10000108
A_POS_Z = 10000109
A_DEFEATED = 50000007
A_TURRET_YAW = 10000111
A_TURRET_PITCH = 10000112
A_MATCH_STATUS = 80000005
A_MAP_PTR = 1000001
A_IS_AI = 50000088
A_MOVE_MODE = 50000089
A_TARGET_MODE = 50000090
A_FIRE_RATE = 50000091

PHASES = [
    # (name, duration_s, applySettings camera payload or None for stock).
    # Exposure lesson from round 1: darker = worse for the NN (stock 16.1% >>
    # -3EV 5.1% >> -5EV 0.2%) — the model needs the armor PATTERN visible, not
    # just lightbars. User-calibrated: 1/120 shutter ~ right for traditional CV;
    # slightly brighter again for the NN.
    ("stock", 30, None),
    ("s120", 210, {"enabled": 1, "fovDegrees": 45, "shutterSpeed": 120}),
]


class GameLink:
    """WS client: telemetry cache + console.exec sender."""

    def __init__(self, port):
        self.port = port
        self.ws = ws_connect(f"ws://127.0.0.1:{port}/", max_size=None)
        self.maps = {}
        self.watched = set()
        self.next_id = 950001
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()
        self.watch(list(range(1, 257)))

    def _reconnect(self):
        # The game WS occasionally resets under load / on resolution change.
        # Without reconnect the reader thread dies and the telemetry cache
        # FREEZES at stale values (looks like the turret stuck off-target).
        # Reconnect and re-subscribe every watched map so the cache stays live.
        for _ in range(30):
            try:
                with self.lock:
                    self.ws = ws_connect(f"ws://127.0.0.1:{self.port}/", max_size=None)
                    prev = list(self.watched)
                    self.watched = set()
                self.watch(range(1, 257))
                self.watch(prev)
                print("[GameLink] WS reconnected", flush=True)
                return True
            except Exception:
                time.sleep(1)
        return False

    def _reader(self):
        while True:
            try:
                for raw in self.ws:
                    try:
                        p = json.loads(raw)
                    except Exception:
                        continue
                    if p.get("type") == 0 and p.get("method") == "watchAttributeMaps.result":
                        discovered = []
                        with self.lock:
                            for r in p.get("params", {}).get("watch_attribute_maps_results", []):
                                m = self.maps.setdefault(r["attribute_map_id"], {})
                                for k, v in (r.get("attributes") or {}).items():
                                    m[int(k)] = v
                                    if int(k) == A_MAP_PTR and isinstance(v, (int, float)) and v > 0:
                                        discovered.append(int(v))
                        if discovered:
                            self.watch(discovered)
            except Exception:
                pass
            # socket closed/errored -> reconnect and keep going
            if not self._reconnect():
                return

    def _send(self, method, params):
        with self.lock:
            self.next_id += 1
            msg = {"type": 0, "id": self.next_id, "method": method, "params": params}
            ws = self.ws
        try:
            ws.send(json.dumps(msg))
        except Exception:
            # let the reader thread handle reconnect; drop this one command
            pass

    def watch(self, ids):
        fresh = [i for i in ids if i not in self.watched]
        if not fresh:
            return
        self.watched.update(fresh)
        self._send("attribute.watchAttributeMaps", {"attribute_map_ids": fresh, "watch_type": 2})

    def exec(self, command):
        self._send("console.exec", {"command": command})

    def apply_camera(self, camera):
        self._send("rgbCamera.applySettings", {"camera": camera})

    def attr(self, map_id, attr_id):
        with self.lock:
            return self.maps.get(map_id, {}).get(attr_id)

    def match_status(self):
        with self.lock:
            for m in self.maps.values():
                if A_MATCH_STATUS in m:
                    return m[A_MATCH_STATUS]
        return None

    def find_pid0_map(self):
        with self.lock:
            for mid, m in self.maps.items():
                if m.get(A_PLAYER_ID) == 0 and A_TURRET_YAW in m:
                    return mid
        return None

    def robots_snapshot(self):
        """All vehicle combat maps: pid/team/class/pos/defeated (ground truth for
        frame visibility labeling)."""
        out = []
        with self.lock:
            for mid, m in self.maps.items():
                if A_CLASS not in m or A_POS_X not in m or A_PLAYER_ID not in m:
                    continue
                out.append({
                    "pid": m.get(A_PLAYER_ID), "team": m.get(A_TEAM_ID),
                    "cls": m.get(A_CLASS),
                    "x": m.get(A_POS_X), "y": m.get(A_POS_Y), "z": m.get(A_POS_Z),
                    "dead": 1 if m.get(A_DEFEATED) == 1 else 0,
                })
        return out


def find_game_window():
    """Top-level visible window owned by UnrealEditor/game with a real client area."""
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    best = None

    def exe_of(pid):
        h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value)
            return ""
        finally:
            kernel32.CloseHandle(h)

    def cb(hwnd, _):
        nonlocal best
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        exe = exe_of(pid).lower()
        if not (exe.startswith("unrealeditor") or exe.startswith("gestalt") or exe.startswith("robotbridge")):
            return True
        l, t = win32gui.ClientToScreen(hwnd, (0, 0))
        cr = win32gui.GetClientRect(hwnd)
        w, h = cr[2], cr[3]
        if w < 800 or h < 500:
            return True
        best = (hwnd, title, (l, t, l + w, t + h))
        return True

    win32gui.EnumWindows(cb, None)
    return best


def raise_game_window(hwnd):
    """Bring the game window to the top and pin it there. dxcam grabs a fixed
    SCREEN rectangle, so if the terminal/chat occludes the game rectangle the
    capture silently grabs those pixels instead (looks like the camera
    'reverted'). TOPMOST keeps the game above other windows for the run."""
    HWND_TOPMOST = -1
    SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW = 0x0002, 0x0001, 0x0040
    try:
        win32gui.ShowWindow(hwnd, 9)  # SW_RESTORE
        win32gui.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                              SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--out", default=r"C:\Users\Chclk\Documents\Unreal Projects\gestalt_system\Saved\ext-sentry\capture")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--entity", default="66000012")
    args = ap.parse_args()

    link = GameLink(args.port)
    print(f"[cap] WS connected :{args.port}")

    for _ in range(240):
        if link.match_status() == 1:
            break
        time.sleep(0.5)
    if link.match_status() != 1:
        raise SystemExit("[cap] match never reached running state")

    # View-attach to the REAL blue sentry (RBTakeOver view mode): no duplicate
    # roster body, its built-in AI keeps driving and firing, we just look through
    # its first-person camera.
    sentry_map, sentry_pid = None, None
    for _ in range(60):
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(60000002) == 1004 and m.get(10000036) == 1:
                    sentry_map, sentry_pid = mid, m.get(A_PLAYER_ID)
        if sentry_pid is not None:
            break
        time.sleep(0.5)
    if sentry_pid is None:
        raise SystemExit("[cap] blue sentry not found in telemetry")
    print(f"[cap] blue sentry pid={sentry_pid} map={sentry_map}; view-attach")
    link.exec(f"UEExec RBTakeOver {sentry_pid}")
    time.sleep(2)

    def keeper():
        while True:
            link.exec("BatchSet %d 1 blue sentry" % A_MOVE_MODE)
            link.exec("BatchSet %d 1 blue sentry" % A_TARGET_MODE)
            link.exec("BatchSet %d 25000 blue sentry" % A_FIRE_RATE)
            time.sleep(0.5)

    threading.Thread(target=keeper, daemon=True).start()
    time.sleep(2)

    win = find_game_window()
    if not win:
        raise SystemExit("[cap] game window not found")
    hwnd, title, region = win
    # Crop the bottom strip: the native version footer overlay ("Connected to
    # Steam | Gestalt System - Beta ...") draws there even with -hudhidden=1.
    region = (region[0], region[1], region[2], region[3] - 28)
    print(f"[cap] window '{title}' region={region} (bottom cropped)")
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(1)

    cam = dxcam.create(output_color="BGR")
    my_map = sentry_map  # pose sidecar reads the VIEWED sentry's turret angles
    print(f"[cap] pose sidecar map = {my_map}")

    interval = 1.0 / args.fps
    for name, dur, camera in PHASES:
        if camera:
            link.apply_camera(camera)
            print(f"[cap] applySettings {camera}")
            time.sleep(2)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        sidecar = open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8")
        idx = 0
        t_end = time.time() + dur
        t_next = time.time()
        while time.time() < t_end:
            frame = cam.grab(region=region)
            t = time.time()
            if frame is not None:
                cv2.imwrite(os.path.join(out_dir, f"f{idx:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                sidecar.write(json.dumps({
                    "idx": idx, "t": t,
                    "yaw": link.attr(my_map, A_TURRET_YAW) if my_map else None,
                    "pitch": link.attr(my_map, A_TURRET_PITCH) if my_map else None,
                    "own": {
                        "x": link.attr(my_map, A_POS_X), "y": link.attr(my_map, A_POS_Y),
                        "z": link.attr(my_map, A_POS_Z), "pid": link.attr(my_map, A_PLAYER_ID),
                        "team": link.attr(my_map, A_TEAM_ID),
                    } if my_map else None,
                    "robots": link.robots_snapshot(),
                }) + "\n")
                idx += 1
            t_next += interval
            delay = t_next - time.time()
            if delay > 0:
                time.sleep(delay)
        sidecar.close()
        print(f"[cap] phase={name} frames={idx}")

    print("[cap] done")


if __name__ == "__main__":
    main()
