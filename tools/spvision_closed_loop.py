#!/usr/bin/env python3
"""V3: vision-only closed-loop auto-aim driving a live AI vehicle's turret.

Architecture (the user-specified split):
  - MOVEMENT + DECISION: untouched built-in AI (IsAIControlled=1, strategy layer
    keeps writing AIMoveMode; we only hold AITargetMode=90 so built-in AIMING is
    off and the turret belongs to us).
  - AUTO-AIM: this process. dxcam captures the sentry's first-person view
    (RBTakeOver view-attach), the sp_vision yolo11 OpenVINO detector finds enemy
    armor, and a P-controller converts the pixel error to world yaw/pitch
    commands sent via `UEExec RBExtAim <pid> <yaw> <pitch> [fire]`.
    NO ground-truth positions enter the aim path — telemetry supplies only the
    CURRENT muzzle pose (the encoder reading a real robot would have).

Usage: python tools/spvision_closed_loop.py <wsPort> [--duration 300]
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import dxcam
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vino_detect import preprocess, parse  # noqa: E402
import openvino as ov  # noqa: E402
from game_window_capture import GameLink, find_game_window  # noqa: E402

A_PLAYER_ID = 10000035
A_TEAM_ID = 10000036
A_CLASS = 60000002
A_TURRET_YAW = 10000111
A_TURRET_PITCH = 10000112
A_BULLETS = 63000002
A_HITS = 63000003
A_KILLS = 63000004
A_MATCH_STATUS = 80000005
A_TARGET_MODE = 50000090

FOV_H_DEG = 45.0
FIRE_ERR_PX = 40.0     # fire when armor center within this radius of crosshair
YAW_GAIN = 0.9         # P gain on the angular error
PITCH_GAIN = 0.8
MAX_STEP_DEG = 20.0    # per-command slew clamp
LOST_HOLD_S = 0.6      # keep last aim this long after losing the target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or ".", "assets", "yolo11.xml"))
    ap.add_argument("--score-threshold", type=float, default=0.5)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model if os.path.exists(args.model) else os.path.join(repo, "assets", "yolo11.xml")
    model = ov.Core().compile_model(model_path, "CPU")

    link = GameLink(args.port)
    print(f"[loop] WS connected :{args.port}")

    sentry = pid = None
    for _ in range(240):
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(A_CLASS) == 1004 and m.get(A_TEAM_ID) == 1:
                    sentry, pid = mid, m.get(A_PLAYER_ID)
            running = any(m.get(A_MATCH_STATUS) == 1 for m in link.maps.values())
        if sentry and running:
            break
        time.sleep(0.5)
    if not sentry:
        raise SystemExit("[loop] blue sentry not found")
    print(f"[loop] blue sentry pid={pid} map={sentry}")

    link.exec(f"UEExec RBTakeOver {pid}")
    time.sleep(2)
    # Realistic camera: 45deg FOV + 1/120 shutter (best V1 recall domain).
    link.apply_camera({"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120})
    time.sleep(1.5)
    # Same-cycle strategy override: aiming is OURS for this pid (no keeper races,
    # no built-in AutoAim reclaim windows); movement/decision stay native.
    link.exec(f"ExtAimClaim {pid} 1")
    time.sleep(0.5)

    win = find_game_window()
    if not win:
        raise SystemExit("[loop] game window not found")
    _, title, region = win
    region = (region[0], region[1], region[2], region[3] - 28)
    W = region[2] - region[0]
    H = region[3] - region[1]
    focal_px = (W / 2.0) / math.tan(math.radians(FOV_H_DEG / 2.0))
    print(f"[loop] window '{title}' {W}x{H} focal={focal_px:.0f}px")

    cam = dxcam.create(output_color="BGR")
    b0 = link.attr(sentry, A_BULLETS) or 0
    h0 = link.attr(sentry, A_HITS) or 0
    k0 = link.attr(sentry, A_KILLS) or 0

    frames = 0
    det_frames = 0
    cmds = 0
    last_seen = 0.0
    t_end = time.time() + args.duration
    t_log = 0.0
    # Smooth scan state: a continuous absolute-yaw ramp with velocity FF (the
    # LQR chases a moving setpoint instead of stop-starting on position steps).
    SCAN_RATE_DEG_S = 55.0
    scan_yaw = None
    scan_dir = 1.0
    scan_t_prev = time.time()

    while time.time() < t_end:
        status = None
        with link.lock:
            for m in link.maps.values():
                if A_MATCH_STATUS in m:
                    status = m[A_MATCH_STATUS]
        if status == 2:
            print("[loop] match settled")
            break

        frame = cam.grab(region=region)
        if frame is None:
            time.sleep(0.005)
            continue
        frames += 1
        tensor, scale = preprocess(frame)
        dets = parse(model(tensor)[0], scale, args.score_threshold)
        enemy = [d for d in dets if d["color"] == "red"]

        cur_yaw = link.attr(sentry, A_TURRET_YAW)
        cur_pitch = link.attr(sentry, A_TURRET_PITCH)
        if enemy and cur_yaw is not None:
            det_frames += 1
            last_seen = time.time()
            scan_yaw = None  # tracking resets the scan ramp
            # Nearest-to-crosshair armor.
            cx, cy = W / 2.0, H / 2.0
            best = min(enemy, key=lambda d: (d["box"][0] + d["box"][2] / 2 - cx) ** 2
                       + (d["box"][1] + d["box"][3] / 2 - cy) ** 2)
            ex = (best["box"][0] + best["box"][2] / 2.0) - cx
            ey = (best["box"][1] + best["box"][3] / 2.0) - cy
            dyaw = math.degrees(math.atan2(ex, focal_px)) * YAW_GAIN
            dpitch = -math.degrees(math.atan2(ey, focal_px)) * PITCH_GAIN
            dyaw = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, dyaw))
            dpitch = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, dpitch))
            fire = 1 if (ex * ex + ey * ey) <= FIRE_ERR_PX * FIRE_ERR_PX else 0
            link.exec(f"UEExec RBExtAim {pid} {cur_yaw + dyaw:.2f} {(cur_pitch or 0) + dpitch:.2f} {fire}")
            cmds += 1
        elif time.time() - last_seen > LOST_HOLD_S and cur_yaw is not None:
            # No target: CONTINUOUS scan ramp with velocity feedforward — an
            # absolute setpoint advancing at a fixed rate (independent of the
            # laggy 10Hz telemetry), so the LQR glides instead of step-chasing.
            now = time.time()
            if scan_yaw is None:
                scan_yaw = cur_yaw
                scan_t_prev = now
            scan_yaw += SCAN_RATE_DEG_S * scan_dir * (now - scan_t_prev)
            scan_t_prev = now
            if scan_yaw > 999:
                scan_yaw -= 360
            link.exec(f"UEExec RBExtAim {pid} {scan_yaw:.2f} 0 0 {SCAN_RATE_DEG_S * scan_dir:.1f}")
            cmds += 1
            time.sleep(0.05)

        if time.time() - t_log > 10:
            t_log = time.time()
            print(f"[loop] frames={frames} det_frames={det_frames} cmds={cmds} "
                  f"bullets={link.attr(sentry, A_BULLETS)} hits={link.attr(sentry, A_HITS)} "
                  f"kills={link.attr(sentry, A_KILLS)}")

    link.exec(f"ExtAimClaim {pid} 0")  # release: strategy owns aiming again
    link.exec(f"BatchSet {A_TARGET_MODE} 1 blue sentry")
    summary = {
        "frames": frames, "det_frames": det_frames, "cmds": cmds,
        "bullets": (link.attr(sentry, A_BULLETS) or 0) - b0,
        "hits": (link.attr(sentry, A_HITS) or 0) - h0,
        "kills": (link.attr(sentry, A_KILLS) or 0) - k0,
    }
    summary["hit_rate"] = round(summary["hits"] / summary["bullets"], 3) if summary["bullets"] else None
    print("V3 SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
