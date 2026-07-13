#!/usr/bin/env python3
"""Fixed benchmark: external (vision + PnP + ballistics) aim destroys the RED
OUTPOST in prep stage with the sentry's 300-round allowance.

Scenario (all facts file:line-anchored in Docs/AI/external-sentry-control/):
  - Launch: ai-match-selftest.ps1 -SkipMatchStart 1 (prep stage, status 0 —
    no AI robots, no coach, outpost damageable, supply reload disabled).
  - Spawn blue sentry as pid0 (Respawn 0 66000012 1): CanOperate=1,
    allowance Ammo17mmCount=300 (the true budget), physical 1000.
  - Drive: IsAIControlled=1 + AIMoveMode=60 + AIMoveTargetMarkerId=15
    (elevated standoff (13,-306,25)cm, ~3.95m to the outpost) with
    AITargetMode=0 (exact drive); on arrival hold (AIMoveMode=2) and switch
    AITargetMode=90 (external turret).
  - Aim: window capture -> yolo11 -> red-outpost armor keypoints -> PnP +
    ballistic solve (aim_solver, bullet 24.3 m/s, g=9.81) -> RBExtAim fire.
    Vision-only aim path; fire only while a plate is detected (the rotating
    drum's FixDart plate and gaps waste rounds otherwise).
  - Ground truth comparison (per-shot JSONL): red outpost at
    (-3.81,-2.83,~1.30)m world vs the PnP-solved relative position + the
    naive LOS pitch vs ballistic pitch, quantifying the "shots land low" gap.

Pass condition: outpost Health 1500 -> 0 (75 damaging hits x 20) within the
300-round allowance.

Usage: python tools/outpost_benchmark.py <wsPort> [--marker 15] [--preset s120]
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
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
    "health": 10000003, "health_max": 60000004,
    "allow": 10000033, "real": 10000031, "bullets": 63000002,
    "is_ai": 50000088, "move_mode": 50000089, "target_mode": 50000090,
    "marker": 50000097, "can_operate": 50000021,
    "g_outpost_red": 80002000,
}

RED_OUTPOST_POS_M = (-3.81, -2.83, 0.20)   # transform_define.csv Outpost2026_0 (cm/100)
PLATE_RING_Z_M = 1.30                       # ArmorBake plate z ~101-142cm above origin
STANDOFF_MARKER = 15                        # (13,-306,25)cm elevated perch
STANDOFF_POS_CM = (13.0, -306.0)
FOV_H_DEG = 45.0

CAMERA_PRESETS = {
    "stock": None,
    "s120": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120},
    "s120b": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120, "exposureBias": 1},
    "s60": {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 60},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--marker", type=int, default=STANDOFF_MARKER)
    ap.add_argument("--preset", default="s120", choices=sorted(CAMERA_PRESETS))
    ap.add_argument("--score-threshold", type=float, default=0.4)
    ap.add_argument("--timeout", type=int, default=800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--debug-frames", type=int, default=40,
                    help="save every Nth frame annotated (0=off)")
    ap.add_argument("--mode", default="vision", choices=["vision", "gtsweep"],
                    help="gtsweep: ground-truth aim, pitch-offset sweep -3..+3 "
                         "deg, 20 timed rounds per bucket — separates aim error "
                         "from drum-timing/occlusion caps")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = ov.Core().compile_model(os.path.join(repo, "assets", "yolo11.xml"), "CPU")

    out_path = args.out or os.path.join(
        repo, f"outpost_benchmark_{args.preset}.jsonl")
    log_fp = open(out_path, "w", encoding="utf-8")

    def record(kind, **data):
        log_fp.write(json.dumps({"t": time.time(), "kind": kind, **data}) + "\n")
        log_fp.flush()
        if kind != "shot":
            print(f"[bench] {kind} {json.dumps(data)}")

    link = GameLink(args.port)
    print(f"[bench] WS connected :{args.port}")
    time.sleep(3)

    # --- spawn blue sentry as pid0 (prep stage has no robots) ---
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
        raise SystemExit("[bench] sentry spawn failed")
    record("spawned", map=sentry,
           allow=link.attr(sentry, A["allow"]), real=link.attr(sentry, A["real"]))

    # --- find red outpost map (global var 80002000 -> map id) ---
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
    if not outpost:  # fallback: the 1500-HP building map
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(A["health_max"]) == 1500 and A["pid"] not in m:
                    outpost = mid
    if not outpost:
        raise SystemExit("[bench] red outpost map not found")
    time.sleep(1)
    hp0 = link.attr(outpost, A["health"]) or 1500
    record("outpost", map=outpost, hp=hp0)

    # --- drive to the standoff perch (exact drive), then hold ---
    link.exec("BatchSet %d 1 blue sentry" % A["is_ai"])
    link.exec("BatchSet %d 0 blue sentry" % A["target_mode"])
    link.exec("BatchSet %d %d blue sentry" % (A["marker"], args.marker))
    link.exec("BatchSet %d 60 blue sentry" % A["move_mode"])
    arrived = False
    for _ in range(160):
        x, y = link.attr(sentry, A["px"]), link.attr(sentry, A["py"])
        if x is not None and math.hypot(x - STANDOFF_POS_CM[0], y - STANDOFF_POS_CM[1]) < 60:
            arrived = True
            break
        time.sleep(0.5)
    record("drive", arrived=arrived,
           pos=[link.attr(sentry, A["px"]), link.attr(sentry, A["py"])])
    link.exec("BatchSet %d 2 blue sentry" % A["move_mode"])   # hold position
    link.exec("BatchSet %d 90 blue sentry" % A["target_mode"])  # external turret
    time.sleep(1)

    # --- camera preset + window ---
    preset = CAMERA_PRESETS[args.preset]
    if preset:
        link.apply_camera(preset)
        time.sleep(1.5)
    win = find_game_window()
    if not win:
        raise SystemExit("[bench] game window not found")
    hwnd, _, region = win
    raise_game_window(hwnd)   # pin the game on top — dxcam grabs a screen rect
    time.sleep(1)
    region = (region[0], region[1], region[2], region[3] - 28)
    W, H = region[2] - region[0], region[3] - region[1]
    K = aim_solver.camera_matrix(W, H, FOV_H_DEG)
    cam = dxcam.create(output_color="BGR")
    record("camera", preset=args.preset, w=W, h=H)

    if args.mode == "gtsweep":
        sx = (link.attr(sentry, A["px"]) or 0) / 100.0
        sy = (link.attr(sentry, A["py"]) or 0) / 100.0
        sz = (link.attr(sentry, A["pz"]) or 0) / 100.0
        gx = RED_OUTPOST_POS_M[0] - sx
        gy = RED_OUTPOST_POS_M[1] - sy
        gd = math.hypot(gx, gy)
        gt_yaw = math.degrees(math.atan2(gy, gx))
        buckets = []
        for off10 in range(-30, 35, 5):   # -3.0 .. +3.0 deg, 0.5 steps
            off = off10 / 10.0
            gz = PLATE_RING_Z_M - (sz + 0.45)
            ball = aim_solver.solve_launch_pitch(24.3, gd, gz)
            pitch_cmd = math.degrees(ball[0]) + off if ball else 10 + off
            hp_before = link.attr(outpost, A["health"]) or 0
            for _ in range(20):
                link.exec(f"UEExec RBExtAim 0 {gt_yaw:.2f} {pitch_cmd:.2f} 1")
                time.sleep(0.35)
            time.sleep(1.0)
            hp_after = link.attr(outpost, A["health"]) or 0
            b = {"offset": off, "pitch_cmd": round(pitch_cmd, 2),
                 "hits": int((hp_before - hp_after) / 20),
                 "allow": link.attr(sentry, A["allow"])}
            buckets.append(b)
            record("gtsweep_bucket", **b)
            if (link.attr(sentry, A["allow"]) or 0) <= 0:
                break
        record("gtsweep_summary", dist=round(gd, 2), yaw=round(gt_yaw, 2),
               buckets=buckets)
        log_fp.close()
        print("GTSWEEP DONE " + json.dumps(buckets))
        return

    # point the turret at the outpost to begin (one GT bootstrap sweep is
    # allowed — a real robot would be driven there facing the target too)
    sx = (link.attr(sentry, A["px"]) or 0) / 100.0
    sy = (link.attr(sentry, A["py"]) or 0) / 100.0
    boot_yaw = math.degrees(math.atan2(RED_OUTPOST_POS_M[1] - sy, RED_OUTPOST_POS_M[0] - sx))
    link.exec(f"UEExec RBExtAim 0 {boot_yaw:.1f} 10 0")
    time.sleep(1.5)

    # --- benchmark loop: vision-only aim, GT compared per shot ---
    debug_dir = os.path.join(repo, f"outpost_debug_{args.preset}")
    if args.debug_frames:
        os.makedirs(debug_dir, exist_ok=True)
    from collections import deque
    dist_window = deque(maxlen=5)   # rolling median steadies PnP range jitter
    yaw_window = deque(maxlen=15)   # plate yaws oscillate around the drum axis
    pitch_window = deque(maxlen=15)
    offset_hist = deque(maxlen=40)  # (t, quad aspect) ~2s ≈ 2.4 plate periods
    DRUM_OMEGA = 144.0              # deg/s — rulebook outpost spin (sign unknown)
    last_cmd = None                # (yaw, pitch) last commanded
    last_det_t = 0.0
    t_end = time.time() + args.timeout
    frames = det_frames = shots_cmd = 0
    # Game teardown (selftest observe window ending) closes the WS mid-loop;
    # catch it so the summary still gets computed from the last cached attrs.
    try:
      while time.time() < t_end:
        allow = link.attr(sentry, A["allow"]) or 0
        hp = link.attr(outpost, A["health"]) or 0
        if hp <= 0 or allow <= 0:
            break
        frame = cam.grab(region=region)
        if frame is None:
            time.sleep(0.005)
            continue
        frames += 1
        if frames % 300 == 0:
            raise_game_window(hwnd)  # re-pin in case focus was stolen
        tensor, scale = preprocess(frame)
        dets = parse(model(tensor)[0], scale, args.score_threshold)
        plates = [d for d in dets if d["name"] == "outpost" or
                  (d["color"] == "red" and d["name"] not in ("base",))]
        cur_yaw = link.attr(sentry, A["tyaw"])
        cur_pitch = link.attr(sentry, A["tpitch"])
        if args.debug_frames and frames % args.debug_frames == 0:
            dbg = frame.copy()
            for d in dets:
                pts = np.array(d["keypoints"], np.int32)
                cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
                cv2.putText(dbg, f'{d["color"]}-{d["name"]} {d["confidence"]:.2f}',
                            (int(pts[0][0]), max(0, int(pts[0][1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(debug_dir, f"f{frames:06d}.jpg"), dbg)
        if not plates or cur_yaw is None:
            # Scan fallback: sweep slowly around the bootstrap bearing so the
            # rotating drum's plates re-enter the crosshair region.
            if time.time() - last_det_t > 2.0 and cur_yaw is not None:
                sweep = boot_yaw + 8.0 * math.sin(time.time() * 0.8)
                link.exec(f"UEExec RBExtAim 0 {sweep:.2f} 10 0 "
                          f"{8.0 * 0.8 * math.cos(time.time() * 0.8):.1f}")
                time.sleep(0.05)
            continue
        det_frames += 1
        last_det_t = time.time()
        cx, cy = W / 2.0, H / 2.0
        best = min(plates, key=lambda d: (d["box"][0] + d["box"][2] / 2 - cx) ** 2
                   + (d["box"][1] + d["box"][3] / 2 - cy) ** 2)
        sol = aim_solver.solve_aim(best["keypoints"], K, cur_yaw, cur_pitch,
                                   armor_name=best["name"])
        if not sol:
            continue
        # Rolling-median range: the standoff is fixed, so PnP range jitter is
        # noise; re-solve the ballistic pitch on the median distance.
        dist_window.append(sol["distance_m"])
        med_d = sorted(dist_window)[len(dist_window) // 2]
        if abs(med_d - sol["distance_m"]) > 1e-6:
            z = sol["target_rel_ue"][2] * (med_d / sol["distance_m"])
            solved2 = aim_solver.solve_launch_pitch(24.3, med_d, z)
            if solved2:
                # keep the solver's static offset when re-solving on median range
                sol["pitch_deg"] = math.degrees(solved2[0]) + aim_solver.PITCH_OFFSET_DEG
                sol["distance_m"] = med_d
        # Hard sanity clamp: a ground sentry shooting a 1.3m-high structure at
        # a few meters never needs pitch outside [0, 25] deg — garbage solves
        # that survive the reprojection gate must not run the turret away.
        sol["pitch_deg"] = max(0.0, min(25.0, sol["pitch_deg"]))
        # ground truth from telemetry (NOT used for aiming)
        sx = (link.attr(sentry, A["px"]) or 0) / 100.0
        sy = (link.attr(sentry, A["py"]) or 0) / 100.0
        sz = (link.attr(sentry, A["pz"]) or 0) / 100.0
        muzzle_z = sz + 0.45  # approx turret pivot height above chassis origin
        gx = RED_OUTPOST_POS_M[0] - sx
        gy = RED_OUTPOST_POS_M[1] - sy
        gz = PLATE_RING_Z_M - muzzle_z
        gd = math.hypot(gx, gy)
        gt_los_pitch = math.degrees(math.atan2(gz, gd))
        gt_ball = aim_solver.solve_launch_pitch(24.3, gd, gz)
        # ROTATING-DRUM PHASE FIRE: plates pass the aim line every 120deg/144deg/s
        # = 0.833s. Each plate is only visible ~0.2s, so per-plate drift tracking
        # is hopeless — instead cluster detections into visibility EPISODES,
        # take the episode center as the plate-crossing time, extrapolate the
        # period, and pull the trigger so the bullet ARRIVES at the next
        # crossing: (now + fly_time) mod period ~ next_center.
        PERIOD = 120.0 / DRUM_OMEGA  # 0.833s
        yaw_window.append(sol["yaw_deg"])
        pitch_window.append(sol["pitch_deg"])
        aim_yaw = sorted(yaw_window)[len(yaw_window) // 2]
        aim_pitch = sorted(pitch_window)[len(pitch_window) // 2]
        now = time.time()
        # Drum phase from the plate's FACING angle: the detected quad's aspect
        # ratio (width/height in px) peaks when a plate is frontal. Detection is
        # near-continuous (~86% of frames, oblique plates included), so
        # visibility gaps carry no phase — aspect peaks do. Anchor the last
        # frontal moment, extrapolate the 0.833s plate period, and fire so the
        # bullet ARRIVES at the next frontal crossing (plates occupy only ~23%
        # of the front arc; off-phase rounds hit the bare drum for zero damage).
        # Drum phase from the plate center's PIXEL X-OFFSET against the (fixed)
        # aim line: as the drum spins the plate center sweeps ±r = ±0.276m ≈
        # ±100px across the axis — a huge-SNR periodic signal (aspect ratio was
        # too noisy). Single-bin DFT at the KNOWN plate period locks the phase;
        # the frontal transit is the sweep's zero-crossing, whose side is set by
        # the sweep direction. Fire so the bullet ARRIVES at the crossing.
        # FRONTAL-WIDTH FIRE GATE: a hit only DAMAGES when the plate is frontal
        # (perpendicular impact clears the 1200 speed threshold; oblique hits
        # score zero). Apparent plate width w_px ∝ cos(facing_angle) peaks at
        # frontal, so fire on the rising edge just before the peak — at 144°/s
        # the plate turns ~24° during the 0.17s flight (cos24°=0.91).
        # HOLD aim on the drum axis (median solved yaw/pitch): keeps the camera
        # on the plates (~97% detection) and the turret SETTLED, so fire timing
        # is clean. Chasing the moving plate defeats the settle gate (verified:
        # 21.7% held-axis vs 5.4% chasing).
        aim_yaw = sorted(yaw_window)[len(yaw_window) // 2]
        aim_pitch = sorted(pitch_window)[len(pitch_window) // 2]
        settled = (last_cmd is not None
                   and abs((cur_yaw - last_cmd[0] + 180) % 360 - 180) < 3.0
                   and abs((cur_pitch or 0) - last_cmd[1]) < 2.0)
        centered = abs(best["box"][0] + best["box"][2] / 2.0 - W / 2.0) < 110
        # FRONTAL-WIDTH FIRE GATE (best-known controller: 71/75 hits, HP→80).
        # Damage requires a frontal hit (perpendicular impact clears the 1200
        # speed gate); apparent width w_px ∝ cos(facing) peaks at frontal, so
        # fire on the rising edge just before the peak — at 144°/s the plate
        # turns ~24° during the 0.17s flight, so w_norm≈0.90 while RISING lands
        # the round near-perpendicular. Slow-decay running max keeps w_norm
        # calibrated. (Tried: geometric-lead 2.5%, windowed max 18%, chasing
        # 5% — the width proxy is the most robust across the drum RNG.)
        kp = np.asarray(best["keypoints"], np.float64)
        w_px = (np.linalg.norm(kp[0] - kp[1]) + np.linalg.norm(kp[3] - kp[2])) / 2
        main.w_max = max(w_px, getattr(main, "w_max", w_px) * 0.999)
        w_norm = w_px / max(main.w_max, 1e-3)
        offset_hist.append((now, w_px))
        rising = len(offset_hist) >= 2 and offset_hist[-1][1] >= offset_hist[-2][1]
        fire = 1 if (settled and centered and w_norm > 0.88 and rising) else 0
        shots_cmd += fire
        last_cmd = (aim_yaw, aim_pitch)
        link.exec(f"UEExec RBExtAim 0 {aim_yaw:.2f} {aim_pitch:.2f} {fire}")
        record("shot", solved={k: (round(v, 3) if isinstance(v, float) else
                                   [round(q, 3) for q in v])
                               for k, v in sol.items()},
               gt={"dist": round(gd, 3), "rel": [round(gx, 3), round(gy, 3), round(gz, 3)],
                   "los_pitch": round(gt_los_pitch, 2),
                   "ball_pitch": round(math.degrees(gt_ball[0]), 2) if gt_ball else None,
                   "yaw": round(math.degrees(math.atan2(gy, gx)), 2)},
               turret={"yaw": round(cur_yaw, 2), "pitch": round(cur_pitch or 0, 2)},
               hp=hp, allow=allow, conf=round(best["confidence"], 2), cls=best["name"])
    except Exception as e:
        record("teardown", error=str(e)[:200])

    hp_final = link.attr(outpost, A["health"]) or 0
    allow_final = link.attr(sentry, A["allow"]) or 0
    summary = {
        "preset": args.preset, "frames": frames, "det_frames": det_frames,
        "shots_commanded": shots_cmd,
        "bullets_used": 300 - allow_final,
        "damaging_hits": int((hp0 - hp_final) / 20),
        "hp_final": hp_final,
        "destroyed": hp_final <= 0,
        "hit_rate": round((hp0 - hp_final) / 20 / max(300 - allow_final, 1), 3),
    }
    record("summary", **summary)
    log_fp.close()
    print("BENCH RESULT=" + ("PASS" if summary["destroyed"] else "PARTIAL")
          + " " + json.dumps(summary))


if __name__ == "__main__":
    main()
