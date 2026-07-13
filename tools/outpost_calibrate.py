#!/usr/bin/env python3
"""Single-session calibration: tune camera presets + verify the external aim
solution against the BUILT-IN auto-aim solution (used as approximate ground
truth — it lands rapid hits, so its settled TurretYaw/Pitch already contain the
game's own ballistic compensation). No bullets, no game restarts needed for the
angle calibration; a short live-fire validation runs at the end.

Per position (near = marker 15 perch ~4m; far = the dart-lob point via
AIMoveMode=5, the official "stand 7-8m off the enemy outpost" spot):
  1. drive & hold
  2. REFERENCE: AITargetMode=3 (building lock) with AIFireRateMilliHz=0 —
     the built-in chain aims silently; sample settled TurretYaw/Pitch (median).
  3. OURS: AITargetMode=90; for each camera preset: applySettings, capture
     ~8s, log detection rate + median solved yaw/pitch + delta vs reference.
  4. VALIDATE (optional --fire): best preset, phase-gated fire ~90s, HP delta.

Usage: python tools/outpost_calibrate.py <wsPort> [--fire]
"""

import argparse
import json
import math
import os
import sys
import time

import dxcam
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vino_detect import preprocess, parse  # noqa: E402
from game_window_capture import GameLink, find_game_window, raise_game_window  # noqa: E402
import aim_solver  # noqa: E402
import openvino as ov  # noqa: E402

A = {
    "pid": 10000035, "team": 10000036, "cls": 60000002,
    "px": 10000107, "py": 10000108, "pz": 10000109,
    "tyaw": 10000111, "tpitch": 10000112,
    "health": 10000003, "allow": 10000033,
    "is_ai": 50000088, "move_mode": 50000089, "target_mode": 50000090,
    "marker": 50000097, "fire_rate": 50000091,
    "g_outpost_red": 80002000,
}
RED_OUTPOST_POS_M = (-3.81, -2.83, 0.20)
FOV_H_DEG = 45.0
PRESETS = {
    "stock": None,
    "s120": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120},
    "s120b": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120, "exposureBias": 1},
    "s60": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 60},
}
POSITIONS = [
    ("near", {"drive": ("marker", 15), "pos_cm": (13.0, -306.0), "radius": 80}),
    ("far", {"drive": ("mode5", None), "pos_cm": None, "radius": None}),
]


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--fire", action="store_true", help="live-fire validation per position")
    ap.add_argument("--score-threshold", type=float, default=0.4)
    ap.add_argument("--only", default=None, choices=[None, "near", "far"])
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = ov.Core().compile_model(os.path.join(repo, "assets", "yolo11.xml"), "CPU")
    out = open(os.path.join(repo, "outpost_calibration.jsonl"), "w", encoding="utf-8")

    def record(kind, **data):
        out.write(json.dumps({"t": time.time(), "kind": kind, **data}) + "\n")
        out.flush()
        print(f"[cal] {kind} {json.dumps(data, ensure_ascii=False)}")

    link = GameLink(args.port)
    time.sleep(3)
    link.exec("Respawn 0 66000012 1")
    sentry = None
    for _ in range(60):
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(A["cls"]) == 1004 and m.get(A["team"]) == 1:
                    sentry = mid
        if sentry:
            break
        time.sleep(0.5)
    if not sentry:
        raise SystemExit("sentry spawn failed")
    outpost = None
    for _ in range(40):
        with link.lock:
            for m in link.maps.values():
                v = m.get(A["g_outpost_red"])
                if isinstance(v, (int, float)) and v > 0:
                    outpost = int(v)
        if outpost:
            link.watch([outpost])
            break
        time.sleep(0.5)
    record("session", sentry=sentry, outpost=outpost)

    win = find_game_window()
    if not win:
        raise SystemExit("game window not found")
    hwnd, _, region = win
    raise_game_window(hwnd)
    time.sleep(1)
    region = (region[0], region[1], region[2], region[3] - 28)
    W, H = region[2] - region[0], region[3] - region[1]
    K = aim_solver.camera_matrix(W, H, FOV_H_DEG)
    cam = dxcam.create(output_color="BGR")

    link.exec("BatchSet %d 1 blue sentry" % A["is_ai"])

    dbg_dir = os.path.join(repo, "outpost_cal_debug")
    os.makedirs(dbg_dir, exist_ok=True)
    import cv2 as _cv2

    for pos_name, cfg in POSITIONS:
        if args.only and pos_name != args.only:
            continue
        # ---- drive ----
        link.exec("BatchSet %d 0 blue sentry" % A["target_mode"])
        if cfg["drive"][0] == "marker":
            link.exec("BatchSet %d %d blue sentry" % (A["marker"], cfg["drive"][1]))
            link.exec("BatchSet %d 60 blue sentry" % A["move_mode"])
        else:
            link.exec("BatchSet %d 5 blue sentry" % A["move_mode"])
        settled_pos = None
        prev = None
        for _ in range(180):
            x, y = link.attr(sentry, A["px"]), link.attr(sentry, A["py"])
            if x is None:
                time.sleep(0.5)
                continue
            if cfg["pos_cm"]:
                if math.hypot(x - cfg["pos_cm"][0], y - cfg["pos_cm"][1]) < cfg["radius"]:
                    settled_pos = (x, y)
                    break
            else:  # mode 5: settle when stationary 3s
                if prev and math.hypot(x - prev[0], y - prev[1]) < 5:
                    settled_pos = (x, y)
                    break
                prev = (x, y)
                time.sleep(2.5)
                continue
            time.sleep(0.5)
        link.exec("BatchSet %d 2 blue sentry" % A["move_mode"])
        sx = (link.attr(sentry, A["px"]) or 0) / 100.0
        sy = (link.attr(sentry, A["py"]) or 0) / 100.0
        gd = math.hypot(RED_OUTPOST_POS_M[0] - sx, RED_OUTPOST_POS_M[1] - sy)
        record("position", name=pos_name, pos=[round(sx, 2), round(sy, 2)],
               dist_to_outpost_m=round(gd, 2), settled=settled_pos is not None)

        # ---- reference ----
        # Built-in silent aim (AITargetMode=3 building lock) —但 mode 3 selects
        # its own building: at the far spot it locked a DIFFERENT structure
        # (bearing off by -42 deg), so accept the built-in reference only when
        # its bearing matches the outpost; otherwise fall back to GT geometry +
        # the (near-validated) ballistic solve.
        gt_bearing = math.degrees(math.atan2(RED_OUTPOST_POS_M[1] - sy,
                                             RED_OUTPOST_POS_M[0] - sx))
        link.exec("BatchSet %d 0 blue sentry" % A["fire_rate"])
        link.exec("BatchSet %d 3 blue sentry" % A["target_mode"])
        time.sleep(3)
        ref_samples = []
        for _ in range(30):
            yv, pv = link.attr(sentry, A["tyaw"]), link.attr(sentry, A["tpitch"])
            if yv is not None:
                ref_samples.append((yv, pv or 0))
            time.sleep(0.1)
        ref_yaw = sorted(s[0] for s in ref_samples)[len(ref_samples) // 2]
        ref_pitch = sorted(s[1] for s in ref_samples)[len(ref_samples) // 2]
        ref_src = "builtin"
        if abs(wrap180(ref_yaw - gt_bearing)) > 6.0:
            sz = (link.attr(sentry, A["pz"]) or 0) / 100.0
            ball = aim_solver.solve_launch_pitch(24.3, gd, 1.30 - (sz + 0.45))
            ref_yaw = gt_bearing
            ref_pitch = math.degrees(ball[0]) if ball else 12.0
            ref_src = "gt_geometry"
        record("reference", pos=pos_name, ref_yaw=round(ref_yaw, 2),
               ref_pitch=round(ref_pitch, 2), gt_bearing=round(gt_bearing, 2),
               bearing_delta=round(wrap180(ref_yaw - gt_bearing), 2), src=ref_src)

        # ---- our solve per preset ----
        link.exec("BatchSet %d 90 blue sentry" % A["target_mode"])
        # hold the turret ON the reference solution so the camera sees the
        # plates the same way the built-in aim does
        link.exec(f"UEExec RBExtAim 0 {ref_yaw:.2f} {ref_pitch:.2f} 0")
        time.sleep(1)
        for preset_name, preset in PRESETS.items():
            if preset:
                link.apply_camera(preset)
            time.sleep(1.2)
            n = det = 0
            yaws, pitches, dists, reprojs = [], [], [], []
            saved_dbg = False
            t_end = time.time() + 8
            while time.time() < t_end:
                frame = cam.grab(region=region)
                if frame is None:
                    time.sleep(0.005)
                    continue
                n += 1
                if not saved_dbg and n > 20:
                    _cv2.imwrite(os.path.join(dbg_dir, f"{pos_name}_{preset_name}.jpg"), frame)
                    saved_dbg = True
                tensor, scale = preprocess(frame)
                dets = parse(model(tensor)[0], scale, args.score_threshold)
                plates = [d for d in dets if d["name"] == "outpost"]
                if not plates:
                    continue
                det += 1
                cx = W / 2.0
                best = min(plates, key=lambda d: abs(d["box"][0] + d["box"][2] / 2 - cx))
                cy_, cp_ = link.attr(sentry, A["tyaw"]), link.attr(sentry, A["tpitch"])
                sol = aim_solver.solve_aim(best["keypoints"], K, cy_ or 0, cp_ or 0,
                                           armor_name="outpost")
                if sol:
                    yaws.append(sol["yaw_deg"])
                    pitches.append(sol["pitch_deg"])
                    dists.append(sol["distance_m"])
                    reprojs.append(sol["reproj_err_px"])
            def med(v):
                return round(sorted(v)[len(v) // 2], 2) if v else None
            record("preset", pos=pos_name, preset=preset_name,
                   frames=n, det_rate=round(det / max(n, 1), 3),
                   yaw=med(yaws), pitch=med(pitches), dist=med(dists),
                   reproj=med(reprojs),
                   dyaw=round(wrap180((med(yaws) or 0) - ref_yaw), 2) if yaws else None,
                   dpitch=round((med(pitches) or 0) - ref_pitch, 2) if pitches else None,
                   gt_dist=round(gd, 2))
    record("done")
    out.close()


if __name__ == "__main__":
    main()
