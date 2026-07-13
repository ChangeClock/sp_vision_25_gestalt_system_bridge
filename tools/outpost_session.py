#!/usr/bin/env python3
"""Single long-lived session controller for outpost external-aim tuning.

Launch the game ONCE (prep stage, long keep-alive) separately; this connects and
stays up, reading a hot-reloadable command file so I can drive the sentry to the
NEAR or FAR point, change camera params, switch modes, and hot-tune tracker/aimer
params WITHOUT restarting the game — the /goal "先不反复开关游戏" requirement.

Command file (JSON, re-read every loop): <repo>/session_cmd.json
  {
    "position": "near" | "far",         # drive there when it changes
    "camera":   {applySettings payload},# apply when it changes
    "mode":     "idle" | "validate" | "track",
    "params":   { ...aimer/tracker overrides... },
    "fire":     true/false,             # allow trigger in track mode
    "quit":     false
  }
Status file (JSON, written every ~0.5s): <repo>/session_status.json
  { position, dist_m, det_rate, mode, ekf:{converged,omega,radius},
    validate:{n,dyaw_med,dpitch_med}, track:{fire,hp,allow,bullets} }

Modes:
  validate — game's built-in aim (AITargetMode=3) points the turret at the
             outpost; per frame compare internal TurretYaw/Pitch vs sp_vision's
             external PnP+ballistic solve. Proves camera+solve alignment.
  track    — external turret (AITargetMode=90); run the ported sp_vision
             tracker+aimer+shooter (spvision_tracker) and drive RBExtAim.
"""

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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD_PATH = os.path.join(REPO, "session_cmd.json")
STATUS_PATH = os.path.join(REPO, "session_status.json")

A = {
    "pid": 10000035, "team": 10000036, "cls": 60000002,
    "px": 10000107, "py": 10000108, "pz": 10000109,
    "chassis_yaw": 10000110,
    "tyaw": 10000111, "tpitch": 10000112,
    "health": 10000003, "allow": 10000033, "real": 10000031,
    "is_ai": 50000088, "move_mode": 50000089, "target_mode": 50000090,
    "marker": 50000097, "fire_rate": 50000091,
    "g_outpost_red": 80002000,
    "invincible": 50000013, "defeated": 50000007,
}
RED_OUTPOST_POS_M = (-3.81, -2.83, 0.20)
PLATE_RING_Z_M = 1.30
FOV_H_DEG = 45.0
CROP_BOTTOM = 28   # version footer strip

POSITIONS = {
    # near = elevated standoff perch (marker 15, ~4m); far = drive toward the
    # outpost (mode 5) and STOP at a fixed ~6m standoff. Stopping by distance-to-
    # outpost (known world pos) makes the far point DETERMINISTIC — mode 5 alone
    # snags at random distances (5.8-14.5m observed), which is unusable for a
    # repeatable test.
    "near": {"kind": "marker", "marker": 15, "pos_cm": (13.0, -306.0)},
    # far = the dart-lob STANDOFF near blue spawn (mode 5), firing ACROSS the
    # field at the distant red outpost (~14m). This is the real attack point —
    # blue does NOT cross to the enemy outpost. Stop when the sentry settles.
    "far": {"kind": "mode5", "settle": True},
}
DEFAULT_CAMERA = {"enabled": 1, "fovDegrees": FOV_H_DEG, "shutterSpeed": 120, "iso": 600}


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def read_cmd():
    try:
        with open(CMD_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_status(d):
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:
        pass


def main():
    port = int(sys.argv[1])
    link = GameLink(port)
    print(f"[session] WS connected :{port}", flush=True)
    time.sleep(3)

    link.exec("Respawn 0 66000012 1")
    # The match spawns its OWN blue sentry, so after Respawn there are TWO blue
    # sentries (class 1004, team 1). RBExtAim/drive all target pid 0 (my spawn),
    # so the attr map I READ must also be the pid-0 entity — else I read a frozen
    # ammo-0 sentry while driving a different one. Disambiguate by pid==0.
    sentry = None
    for _ in range(60):
        cands = []
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(A["cls"]) == 1004 and m.get(A["team"]) == 1:
                    cands.append((mid, m.get(A["pid"])))
        pid0 = [mid for mid, p in cands if p == 0]
        if pid0:
            sentry = pid0[0]
            if len(cands) > 1:
                print(f"[session] {len(cands)} blue sentries; picked pid0 map={sentry}", flush=True)
            break
        if cands and _ >= 20:   # pid attr never resolved to 0 -> fall back
            sentry = cands[0][0]
            print(f"[session] WARN no pid0 among {len(cands)} sentries; using map={sentry} pid={cands[0][1]}", flush=True)
            break
        time.sleep(0.5)
    if not sentry:
        raise SystemExit("[session] sentry(pid0) spawn failed")
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
    print(f"[session] sentry map={sentry} outpost map={outpost}", flush=True)
    link.exec("BatchSet %d 1 blue sentry" % A["is_ai"])
    # Make the sentry INVINCIBLE: during prep the AI sentry can be killed by red
    # units and RESPAWN as a new entity (new attr-map id), which strands the
    # cached map and breaks the aim-feedback read. Invincible = stable identity.
    link.exec("BatchSet %d 1 blue sentry" % A["invincible"])

    def resolve_sentry(prev):
        # entity id can change (each Respawn spawns a NEW sentry; old ones linger
        # with a FROZEN pid=0 snapshot). Pick the pid0 blue sentry that is ALIVE
        # (Defeated==0, Health>0) so we never latch onto a dead, non-updating
        # entity (which strands the aim-feedback read). Keep last good if none.
        alive, any0 = [], []
        with link.lock:
            for mid, m in link.maps.items():
                if (m.get(A["cls"]) == 1004 and m.get(A["team"]) == 1
                        and m.get(A["pid"]) == 0):
                    any0.append(mid)
                    hp = m.get(A["health"])
                    dead = m.get(A["defeated"])
                    if (hp is None or hp > 0) and not dead:
                        alive.append(mid)
        if alive:
            return alive[-1]
        return any0[-1] if any0 else prev

    # Fair-play boundary: only the rulebook magnitude is known. The random
    # per-round direction must be inferred from the visual time series; no game
    # angular-speed attribute is read or injected into the tracker.

    win = find_game_window()
    if not win:
        raise SystemExit("[session] game window not found")
    hwnd, _, region = win
    raise_game_window(hwnd)
    time.sleep(1)
    full_h = region[3] - region[1]
    region = (region[0], region[1], region[2], region[3] - CROP_BOTTOM)
    W = region[2] - region[0]
    H = region[3] - region[1]
    # K rebuilt whenever the camera FOV changes (narrowing FOV = longer lens =
    # more pixels on a distant target, key at the 8m far point).
    cur_fov = [FOV_H_DEG]

    def build_K(fov):
        return aim_solver.camera_matrix(W, H, fov, cy_px=full_h / 2.0)

    K = build_K(cur_fov[0])
    cam = dxcam.create(output_color="BGR")
    model = ov.Core().compile_model(os.path.join(REPO, "assets", "yolo11.xml"), "CPU")
    print(f"[session] window {W}x{H} (crop {CROP_BOTTOM}) K.cy={K[1,2]:.1f}", flush=True)

    # optional tracker (ported sp_vision); loaded lazily so validate works without it
    pipeline = None

    state = {"position": None, "camera": None, "mode": "idle"}
    val_dyaw, val_dpitch = [], []
    frames = det = 0
    bullets0 = link.attr(sentry, A["allow"]) or 300
    last_status = 0.0
    last_raise = time.time()

    def drive_to(name):
        nonlocal sentry
        cfg = POSITIONS[name]
        # External turret + keep the gimbal LEVEL & chassis-FORWARD during the
        # drive: with AITargetMode=0 the built-in default now faces the velocity
        # vector (turret-follows-travel change), which during tunnel-crouch
        # maneuvering points the gun into the tunnel wall and JAMS the crossing
        # (mirrors AiStrategy.ts:1442's "过洞缩头停索敌→云台回正"). Holding the
        # turret at ChassisYaw + pitch 0 每步 keeps it level so tunnels pass.
        link.exec("BatchSet %d 90 blue sentry" % A["target_mode"])
        link.exec("ExtAimClaim 0 1")
        if cfg["kind"] == "marker":
            link.exec("BatchSet %d %d blue sentry" % (A["marker"], cfg["marker"]))
            link.exec("BatchSet %d 60 blue sentry" % A["move_mode"])
        else:
            link.exec("BatchSet %d 5 blue sentry" % A["move_mode"])
        stuck = 0
        prev = None
        for _ in range(260):
            sentry = resolve_sentry(sentry)
            x, y = link.attr(sentry, A["px"]), link.attr(sentry, A["py"])
            cyaw = link.attr(sentry, A["chassis_yaw"])
            if x is None:
                time.sleep(0.3); continue
            if cyaw is not None:
                link.exec(f"UEExec RBExtAim 0 {cyaw:.1f} 0 0")   # level, forward
            if cfg["kind"] == "marker":
                if math.hypot(x - cfg["pos_cm"][0], y - cfg["pos_cm"][1]) < 70:
                    break
            else:  # mode5: settle at the dart-lob standoff near spawn
                if prev and math.hypot(x - prev[0], y - prev[1]) < 5:
                    stuck += 1
                    if stuck > 14:   # ~5s stationary -> settled at standoff
                        break
                else:
                    stuck = 0
            prev = (x, y); time.sleep(0.35)
        link.exec("BatchSet %d 2 blue sentry" % A["move_mode"])  # hold
        final_d = math.hypot(RED_OUTPOST_POS_M[0] - (link.attr(sentry, A["px"]) or 0) / 100.0,
                             RED_OUTPOST_POS_M[1] - (link.attr(sentry, A["py"]) or 0) / 100.0)
        print(f"[session] {name} stop dist_to_outpost={final_d:.1f}m", flush=True)
        print(f"[session] arrived {name} pos=({link.attr(sentry,A['px']):.0f},"
              f"{link.attr(sentry,A['py']):.0f})", flush=True)

    from collections import deque
    hold_yaw = deque(maxlen=12)   # vision median-hold while EKF not converged
    hold_pitch = deque(maxlen=12)
    last_seen = 0.0

    def coarse_acquire():
        # Point the turret at the outpost's general direction from POSITION
        # (nav-style coarse pointing — a real sentry is driven facing its
        # target). Fine aim + fire remain vision-only (sp_vision). Uses the
        # known outpost world pos ONLY to break the see-nothing/can't-aim
        # deadlock; not used in the aim solution.
        nonlocal sentry
        sentry = resolve_sentry(sentry)
        sxx = (link.attr(sentry, A["px"]) or 0) / 100.0
        syy = (link.attr(sentry, A["py"]) or 0) / 100.0
        szz = (link.attr(sentry, A["pz"]) or 0) / 100.0
        bx, by = RED_OUTPOST_POS_M[0] - sxx, RED_OUTPOST_POS_M[1] - syy
        bd = math.hypot(bx, by)
        bear = math.degrees(math.atan2(by, bx))
        elev = math.degrees(math.atan2(PLATE_RING_Z_M - (szz + 0.45), bd))
        link.exec(f"UEExec RBExtAim 0 {bear:.2f} {elev:.2f} 0")

    print("[session] ready — edit session_cmd.json to drive", flush=True)
    reset_token = [None]

    def do_reset(n):
        # clean-slate a destruction test: refill the sentry's 17mm allowance +
        # physical ammo and restore the outpost to full HP so "N 发下前哨" starts
        # from 1500/full budget. (Buildings HP is host-authoritative; BatchSet
        # writes it directly.)
        link.exec("BatchSet %d %d blue sentry" % (A["allow"], n))
        link.exec("BatchSet %d %d blue sentry" % (A["real"], n))
        link.exec("BatchSet %d 1 blue sentry" % A["invincible"])
        if outpost:
            link.exec("BatchSet %d 1500 red outpost" % A["health"])
        print(f"[session] RESET allowance/real={n}, outpost HP->1500", flush=True)

    while True:
        cmd = read_cmd()
        if cmd.get("quit"):
            break
        sentry = resolve_sentry(sentry)   # entity id can change on respawn
        params = dict(cmd.get("params", {}))
        # hot reset/replenish for a clean N-round test
        rt = cmd.get("reset")
        if rt is not None and rt != reset_token[0]:
            do_reset(int(cmd.get("allowance", 300)))
            reset_token[0] = rt
            bullets0 = int(cmd.get("allowance", 300))
        # hot position change
        pos = cmd.get("position", "near")
        if pos != state["position"] and pos in POSITIONS:
            drive_to(pos)
            state["position"] = pos
            val_dyaw.clear(); val_dpitch.clear()
        # hot camera change
        camera = cmd.get("camera", DEFAULT_CAMERA)
        if camera != state["camera"]:
            link.apply_camera(camera)
            state["camera"] = camera
            new_fov = float(camera.get("fovDegrees", FOV_H_DEG))
            if abs(new_fov - cur_fov[0]) > 1e-3:
                cur_fov[0] = new_fov
                K = build_K(new_fov)
                if pipeline is not None:
                    pipeline.K = K
                print(f"[session] FOV -> {new_fov}, K rebuilt fx={K[0,0]:.0f}", flush=True)
            time.sleep(1.0)
        mode = cmd.get("mode", "idle")
        if mode != state["mode"]:
            if mode == "validate":
                link.exec("BatchSet %d 0 blue sentry" % A["fire_rate"])
                link.exec("BatchSet %d 3 blue sentry" % A["target_mode"])
            elif mode == "track":
                link.exec("BatchSet %d 90 blue sentry" % A["target_mode"])
                link.exec("ExtAimClaim 0 1")
                time.sleep(0.3)
                coarse_acquire()   # start pointed at the outpost
                time.sleep(1.0)
                hold_yaw.clear(); hold_pitch.clear()
            state["mode"] = mode
            val_dyaw.clear(); val_dpitch.clear()

        if time.time() - last_raise > 20:
            raise_game_window(hwnd); last_raise = time.time()

        frame = cam.grab(region=region)
        if frame is None:
            time.sleep(0.005); continue
        frames += 1
        cur_yaw = link.attr(sentry, A["tyaw"])
        cur_pitch = link.attr(sentry, A["tpitch"])
        sx = (link.attr(sentry, A["px"]) or 0) / 100.0
        sy = (link.attr(sentry, A["py"]) or 0) / 100.0
        gd = math.hypot(RED_OUTPOST_POS_M[0] - sx, RED_OUTPOST_POS_M[1] - sy)

        if mode == "idle" or cur_yaw is None:
            _status(link, sentry, outpost, state, gd, frames, det, val_dyaw, val_dpitch,
                    bullets0, last_status)
            time.sleep(0.05)
            continue

        tensor, scale = preprocess(frame)
        dets = parse(model(tensor)[0], scale, cmd.get("score_threshold", 0.4))
        plates = [d for d in dets if d["name"] == "outpost"]

        if mode == "validate":
            if plates:
                det += 1
                cx = W / 2.0
                best = min(plates, key=lambda d: abs(d["box"][0] + d["box"][2] / 2 - cx))
                sol = aim_solver.solve_aim(best["keypoints"], K, cur_yaw, cur_pitch,
                                           armor_name="outpost")
                if sol:
                    val_dyaw.append(wrap180(sol["yaw_deg"] - cur_yaw))
                    val_dpitch.append(sol["pitch_deg"] - cur_pitch)

        elif mode == "track":
            pipeline = _ensure_pipeline(pipeline, K, params)
            out = pipeline.step(plates, cur_yaw, cur_pitch, time.time(),
                                cmd.get("bullet_speed", 24.3)) if pipeline else {}
            if plates:
                det += 1
                last_seen = time.time()
                # push the median-hold window from the nearest-center plate's
                # solved world bearing (vision-only, smoothed)
                cx0 = W / 2.0
                best = min(plates, key=lambda d: abs(d["box"][0] + d["box"][2] / 2 - cx0))
                pose = aim_solver.solve_plate_pose(best["keypoints"], K, cur_yaw,
                                                   cur_pitch, armor_name="outpost")
                if pose:
                    px, py, pz = pose["xyz"]
                    hold_yaw.append(math.degrees(math.atan2(py, px)))
                    hold_pitch.append(math.degrees(math.atan2(pz, math.hypot(px, py))))

            fire = 1 if (out.get("fire") and cmd.get("fire", False)) else 0
            st = out.get("state")
            # HIT-RATE GATE: only ~45% of frames actually detect the plate; the
            # rest are temp_lost (EKF-predicted, blind) shots that mostly miss the
            # small spinning plate. Gate fire to frames with a REAL detection
            # (fire_on_detect), and optionally only when a detected plate's box
            # center is within fire_center_px of image center (= plate at the aim
            # bearing → "coincidence fire", hits the plate as it crosses).
            if fire and params.get("fire_on_detect", True):
                if not plates:
                    fire = 0
                else:
                    fcpx = params.get("fire_center_px", 0)
                    if fcpx:
                        cxc = W / 2.0
                        near = min(abs(d["box"][0] + d["box"][2] / 2 - cxc)
                                   for d in plates)
                        if near > fcpx:
                            fire = 0
            # DRUM-AXIS AIM CLAMP: the outpost is a fixed structure whose plates
            # all sit within radius 0.28m (~3° at 6m) of the spin axis, so a
            # valid aim is never far from the axis bearing (median of recent
            # solved detections). An EKF sign-flip / bad prediction can command
            # a wild yaw that swings the turret off the outpost and loses the
            # lock; clamp the aimer output to ±(clamp_deg) of the axis.
            if out.get("control") and hold_yaw:
                axis_y = sorted(hold_yaw)[len(hold_yaw) // 2]
                axis_p = sorted(hold_pitch)[len(hold_pitch) // 2]
                cl = params.get("aim_clamp_deg", 6.0)
                dy = (out["yaw_deg"] - axis_y + 180) % 360 - 180
                out["yaw_deg"] = axis_y + max(-cl, min(cl, dy))
                out["pitch_deg"] = axis_p + max(-cl, min(cl, out["pitch_deg"] - axis_p))
            if out.get("control") and st in ("tracking", "temp_lost"):
                # The aimer has a VALID target (tracking OR temp_lost — the EKF
                # keeps predicting the rotating plate through the outpost's long
                # occlusion gaps, which is the whole point of the model). Drive
                # the turret to the predicted incoming plate + let the shooter's
                # settle gate pull the trigger. Firing during temp_lost is
                # correct: at 8m the plate is visible only ~12% of frames, so
                # most shots must be at the PREDICTED position.
                link.exec(f"UEExec RBExtAim 0 {out['yaw_deg']:.2f} "
                          f"{out['pitch_deg']:.2f} {fire}")
            elif hold_yaw and time.time() - last_seen < 1.0:
                # no target yet but seeing plates: hold the drum axis (median)
                hy = sorted(hold_yaw)[len(hold_yaw) // 2]
                hp_ = sorted(hold_pitch)[len(hold_pitch) // 2]
                link.exec(f"UEExec RBExtAim 0 {hy:.2f} {hp_:.2f} 0")
            elif time.time() - last_seen > 1.5:
                # lost sight for a while: re-acquire the outpost direction
                coarse_acquire()
                last_seen = time.time() - 0.5

        now = time.time()
        if now - last_status > 0.5:
            last_status = now
            gt_bearing = math.degrees(math.atan2(RED_OUTPOST_POS_M[1] - sy,
                                                 RED_OUTPOST_POS_M[0] - sx))
            diag = {"turret": [round(cur_yaw, 1), round(cur_pitch, 1)],
                    "n_plates": len(plates), "gt_bearing": round(gt_bearing, 1)}
            if plates:
                cxd = W / 2.0
                bd = min(plates, key=lambda d: abs(d["box"][0] + d["box"][2] / 2 - cxd))
                pd = aim_solver.solve_plate_pose(bd["keypoints"], K, cur_yaw, cur_pitch, "outpost")
                if pd:
                    px, py, pz = pd["xyz"]
                    diag["plate"] = {"bearing": round(math.degrees(math.atan2(py, px)), 1),
                                     "dist": round(math.sqrt(px * px + py * py + pz * pz), 2),
                                     "yaw": round(math.degrees(pd["yaw"]), 1)}
            _status(link, sentry, outpost, state, gd, frames, det, val_dyaw, val_dpitch,
                    bullets0, now, pipeline, diag)

    write_status({"done": True})
    print("[session] quit", flush=True)


def _ensure_pipeline(pipeline, K, params):
    if pipeline is not None:
        pipeline.set_params(params)
        return pipeline
    try:
        import spvision_tracker
        p = spvision_tracker.SpvisionAimPipeline(K, params=params)
        print("[session] spvision_tracker loaded", flush=True)
        return p
    except Exception as e:
        print(f"[session] tracker not ready: {e}", flush=True)
        return None


def _status(link, sentry, outpost, state, gd, frames, det, dyaw, dpitch, b0, now,
            pipeline=None, diag=None):
    st = {
        "position": state["position"], "mode": state["mode"],
        "dist_m": round(gd, 2),
        "det_rate": round(det / max(frames, 1), 3),
        "hp": link.attr(outpost, A["health"]),
        "allow": link.attr(sentry, A["allow"]),
        "bullets": (b0 - (link.attr(sentry, A["allow"]) or b0)),
    }
    if dyaw:
        st["validate"] = {
            "n": len(dyaw),
            "dyaw_med": round(float(np.median(dyaw)), 3),
            "dpitch_med": round(float(np.median(dpitch)), 3),
            "dyaw_std": round(float(np.std(dyaw)), 3),
            "dpitch_std": round(float(np.std(dpitch)), 3),
        }
    if pipeline is not None and hasattr(pipeline, "status"):
        st["ekf"] = pipeline.status()
    if diag:
        st.update(diag)
    write_status(st)


if __name__ == "__main__":
    main()
