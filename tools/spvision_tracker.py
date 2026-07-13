#!/usr/bin/env python3
"""Faithful Python port of sp_vision_25's rotating-target auto-aim math, for the
UE-simulator bridge. Ports (file:line into the fork):
  - tools/extended_kalman_filter.cpp      -> EKF (predict/update, Joseph form)
  - tasks/auto_aim/target.cpp             -> Target: 11-state CTRV EKF
      x = [cx,vx,cy,vy,cz,vz, a, w, r, l, h]  (target.cpp:34-39)
      a=tracked-plate yaw, w=angular vel, r=spin radius, l=r2-r1, h=z2-z1
      f(): const-vel center + const-turn a (target.cpp:79-91)
      outpost omega clamp |w|>2 -> ±2.51 (target.cpp:131-133)
      armor_xyza_list / h_armor_xyz / h_jacobian (target.cpp:229-317)
      data-association nearest-by-angle over 3 closest plates (target.cpp:138-185)
  - tasks/auto_aim/tracker.cpp            -> Tracker: lost/detecting/tracking/
      temp_lost state machine, outpost_max_temp_lost_count (tracker.cpp:180-229),
      set_target per-class radius/armor_num/P0 (tracker.cpp:231-264)
  - tasks/auto_aim/aimer.cpp              -> Aimer: flight-time iteration
      (aimer.cpp:78-116) + choose_aim_point coming/leaving, outpost 70/30
      (aimer.cpp:144-208); yaw=atan2(y,x)+off, pitch=-(traj.pitch+off)
  - tasks/auto_aim/shooter.cpp            -> settle gate (shooter.cpp:19-41)
  - tools/trajectory.cpp                  -> closed-form low arc (g param)
  - tools/math_tools.cpp                  -> limit_rad/xyz2ypd/jacobian

UE-bridge adaptations: the "camera" is the gun-view (R_camera2gimbal=I,
t=0, R_gimbal2imubody=I); world frame = gimbal-relative UE world via
aim_solver.ue_cam_to_world(turret_yaw,pitch). Aimer pitch is negated back to
UE up-positive on output. g=9.81 (sim). demo.yaml params exposed for hot-tune.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aim_solver  # noqa: E402

PI = math.pi


def limit_rad(a):
    while a > PI:
        a -= 2 * PI
    while a <= -PI:
        a += 2 * PI
    return a


def xyz2ypd(xyz):
    x, y, z = xyz
    return np.array([math.atan2(y, x),
                     math.atan2(z, math.hypot(x, y)),
                     math.sqrt(x * x + y * y + z * z)])


def xyz2ypd_jacobian(xyz):
    x, y, z = xyz
    r2 = x * x + y * y
    dyaw = [-y / r2, x / r2, 0.0]
    denom = (z * z / r2 + 1)
    dpitch = [-(x * z) / (denom * r2 ** 1.5),
              -(y * z) / (denom * r2 ** 1.5),
              1.0 / (denom * r2 ** 0.5)]
    r3 = (r2 + z * z) ** 0.5
    ddist = [x / r3, y / r3, z / r3]
    return np.array([dyaw, dpitch, ddist])


# --- default params from configs/demo.yaml ---
DEFAULT_PARAMS = {
    "g": 9.81,                    # sim gravity (fork uses 9.7833; sim is 9.81)
    "min_detect_count": 5,        # demo.yaml
    "max_temp_lost_count": 15,
    "outpost_max_temp_lost_count": 75,
    "decision_speed": 8.0,
    "high_speed_delay_time": 0.030,
    "low_speed_delay_time": 0.015,
    "coming_angle_deg": 60.0,     # overridden to 70 for outpost (aimer.cpp:194)
    "leaving_angle_deg": 20.0,    # overridden to 30 for outpost
    "first_tolerance_deg": 3.0,
    "second_tolerance_deg": 2.0,
    "judge_distance": 2.0,
    "yaw_offset_deg": 0.0,        # calibrate vs internal aim in Step 2
    "pitch_offset_deg": 0.0,
    "enemy_color": "red",
    "detector_dt": 0.005,         # aimer.cpp:55 detector-aimer cost
}


def trajectory(v0, d, h, g):
    """trajectory.cpp:9-30 closed-form low arc. Returns (pitch_rad, fly_time) or None."""
    a = g * d * d / (2 * v0 * v0)
    b = -d
    c = a + h
    delta = b * b - 4 * a * c
    if delta < 0:
        return None
    p1 = math.atan((-b + math.sqrt(delta)) / (2 * a))
    p2 = math.atan((-b - math.sqrt(delta)) / (2 * a))
    t1 = d / (v0 * math.cos(p1))
    t2 = d / (v0 * math.cos(p2))
    return (p1, t1) if t1 < t2 else (p2, t2)


class EKF:
    """Generic EKF with additive-with-angle-wrap x_add (ekf.cpp + target x_add)."""

    def __init__(self, x0, P0):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.I = np.eye(len(self.x))
        self.recent_nis_failures = [0]
        self.window_size = 100

    @staticmethod
    def _x_add(a, b):  # target.cpp:43-47 — wrap the angle state x[6]
        c = a + b
        c[6] = limit_rad(c[6])
        return c

    def predict(self, F, Q, f):
        self.P = F @ self.P @ F.T + Q
        self.x = f(self.x)
        return self.x

    def update(self, z, H, R, h, z_subtract):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.P = (self.I - K @ H) @ self.P @ (self.I - K @ H).T + K @ R @ K.T
        self.x = self._x_add(self.x, K @ z_subtract(z, h(self.x)))
        # chi-square NIS bookkeeping (ekf.cpp:59-81) — used by tracker converge gate
        residual = z_subtract(z, h(self.x))
        nis = float(residual.T @ np.linalg.inv(S) @ residual)
        self.recent_nis_failures.append(1 if nis > 0.711 else 0)
        if len(self.recent_nis_failures) > self.window_size:
            self.recent_nis_failures.pop(0)
        return self.x


class Target:
    """11-state CTRV rotating target (target.cpp)."""

    def __init__(self, armor, t, radius, armor_num, P0_dig):
        self.name = armor["name"]
        self.armor_type = armor["type"]
        self.jumped = False
        self.last_id = 0
        self.update_count = 0
        self.armor_num = armor_num
        self.t = t
        self.is_switch = False
        self.is_converged = False
        self.switch_count = 0

        xyz = armor["xyz_in_world"]
        ypr0 = armor["ypr_in_world"][0]
        cx = xyz[0] + radius * math.cos(ypr0)   # target.cpp:30
        cy = xyz[1] + radius * math.sin(ypr0)
        cz = xyz[2]
        x0 = [cx, 0, cy, 0, cz, 0, ypr0, 0, radius, 0, 0]   # target.cpp:39
        P0 = np.diag(np.array(P0_dig, float))
        self.ekf = EKF(x0, P0)
        # omega-clamp params (sp_vision-faithful defaults; overridden per range)
        self.p_omega_clamp_thr = 2.0
        self.p_omega_lock = 2.51
        self.p_omega_early = False

    # --- prediction (target.cpp:75-136) ---
    def predict_dt(self, dt):
        F = np.eye(11)
        for i in (0, 2, 4, 6):   # position/angle <- +vel*dt
            F[i, i + 1] = dt
        if self.name == "outpost":
            v1, v2 = 10.0, 0.1
        else:
            v1, v2 = 100.0, 400.0
        a = dt ** 4 / 4
        b = dt ** 3 / 2
        c = dt ** 2
        Q = np.zeros((11, 11))
        for base in (0, 2, 4):   # x/y/z blocks with v1
            Q[base, base] = a * v1; Q[base, base + 1] = b * v1
            Q[base + 1, base] = b * v1; Q[base + 1, base + 1] = c * v1
        Q[6, 6] = a * v2; Q[6, 7] = b * v2       # angle block with v2
        Q[7, 6] = b * v2; Q[7, 7] = c * v2

        def f(x):
            xp = F @ x
            xp[6] = limit_rad(xp[6])
            return xp

        # Outpost omega clamp (target.cpp:131-133). sp_vision clamps to ±2.51
        # only once converged AND |w|>2. At long range (~8m) detection is too
        # sparse to observe the spin, so w drifts toward 0 and the clamp never
        # fires. The outpost's rate is a KNOWN physical constant, so we snap
        # |w| to it once a few updates land — estimating only the SIGN from the
        # EKF — via tunable params (thr low, lock=2.513). p defaults preserve
        # sp_vision behaviour (thr=2, lock=2.51, gate=convergence).
        if self.name == "outpost":
            thr = self.p_omega_clamp_thr
            gate = (self.update_count > 3) if self.p_omega_early else self.convergened()
            if gate and abs(self.ekf.x[7]) > thr:
                self.ekf.x[7] = self.p_omega_lock if self.ekf.x[7] > 0 else -self.p_omega_lock
        self.ekf.predict(F, Q, f)

    def predict_to(self, t):
        self.predict_dt(t - self.t)
        self.t = t

    # --- measurement geometry (target.cpp:266-317) ---
    def h_armor_xyz(self, x, i):
        angle = limit_rad(x[6] + i * 2 * PI / self.armor_num)
        use_lh = (self.armor_num == 4) and (i in (1, 3))
        r = x[8] + x[9] if use_lh else x[8]
        ax = x[0] - r * math.cos(angle)
        ay = x[2] - r * math.sin(angle)
        az = x[4] + x[10] if use_lh else x[4]
        return np.array([ax, ay, az])

    def armor_xyza_list(self):
        out = []
        for i in range(self.armor_num):
            angle = limit_rad(self.ekf.x[6] + i * 2 * PI / self.armor_num)
            xyz = self.h_armor_xyz(self.ekf.x, i)
            out.append(np.array([xyz[0], xyz[1], xyz[2], angle]))
        return out

    def h_jacobian(self, x, i):
        angle = limit_rad(x[6] + i * 2 * PI / self.armor_num)
        use_lh = (self.armor_num == 4) and (i in (1, 3))
        r = x[8] + x[9] if use_lh else x[8]
        dx_da = r * math.sin(angle)
        dy_da = -r * math.cos(angle)
        dx_dr = -math.cos(angle)
        dy_dr = -math.sin(angle)
        dx_dl = -math.cos(angle) if use_lh else 0.0
        dy_dl = -math.sin(angle) if use_lh else 0.0
        dz_dh = 1.0 if use_lh else 0.0
        H_xyza = np.array([
            [1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl, 0],
            [0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dz_dh],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ], float)
        armor_xyz = self.h_armor_xyz(x, i)
        H_ypd = xyz2ypd_jacobian(armor_xyz)
        H_ypda = np.array([
            [H_ypd[0, 0], H_ypd[0, 1], H_ypd[0, 2], 0],
            [H_ypd[1, 0], H_ypd[1, 1], H_ypd[1, 2], 0],
            [H_ypd[2, 0], H_ypd[2, 1], H_ypd[2, 2], 0],
            [0, 0, 0, 1],
        ], float)
        return H_ypda @ H_xyza

    # --- data association + update (target.cpp:138-223) ---
    def update(self, armor):
        xyza_list = self.armor_xyza_list()
        idx = list(range(self.armor_num))
        idx.sort(key=lambda i: xyz2ypd(xyza_list[i][:3])[2])   # nearest 3 by distance
        best_id, min_err = 0, 1e10
        for i in idx[:3]:
            xyza = xyza_list[i]
            ypd = xyz2ypd(xyza[:3])
            err = abs(limit_rad(armor["ypr_in_world"][0] - xyza[3])) + \
                abs(limit_rad(armor["ypd_in_world"][0] - ypd[0]))
            if abs(err) < abs(min_err):
                best_id, min_err = i, err
        if best_id != 0:
            self.jumped = True
        self.is_switch = best_id != self.last_id
        if self.is_switch:
            self.switch_count += 1
        self.last_id = best_id
        self.update_count += 1
        self._update_ypda(armor, best_id)

    def _update_ypda(self, armor, i):
        H = self.h_jacobian(self.ekf.x, i)
        center_yaw = math.atan2(armor["xyz_in_world"][1], armor["xyz_in_world"][0])
        delta_angle = limit_rad(armor["ypr_in_world"][0] - center_yaw)
        R = np.diag([4e-3, 4e-3,
                     math.log(abs(delta_angle) + 1) + 1,
                     math.log(abs(armor["ypd_in_world"][2]) + 1) / 200 + 9e-2])

        def h(x):
            xyz = self.h_armor_xyz(x, i)
            ypd = xyz2ypd(xyz)
            angle = limit_rad(x[6] + i * 2 * PI / self.armor_num)
            return np.array([ypd[0], ypd[1], ypd[2], angle])

        def z_subtract(a, b):
            c = a - b
            c[0] = limit_rad(c[0]); c[1] = limit_rad(c[1]); c[3] = limit_rad(c[3])
            return c

        ypd = armor["ypd_in_world"]
        ypr = armor["ypr_in_world"]
        z = np.array([ypd[0], ypd[1], ypd[2], ypr[0]])
        self.ekf.update(z, H, R, h, z_subtract)

    def diverged(self):
        r_ok = 0.05 < self.ekf.x[8] < 0.5
        l_ok = 0.05 < self.ekf.x[8] + self.ekf.x[9] < 0.5
        return not (r_ok and l_ok)

    def convergened(self):
        if self.name != "outpost" and self.update_count > 3 and not self.diverged():
            self.is_converged = True
        if self.name == "outpost" and self.update_count > 10 and not self.diverged():
            self.is_converged = True
        return self.is_converged

    def ekf_x(self):
        return self.ekf.x


class Tracker:
    """State machine + per-class target seeding (tracker.cpp)."""

    def __init__(self, params):
        self.p = params
        self.state = "lost"
        self.detect_count = 0
        self.temp_lost_count = 0
        self.last_t = None
        self.target = None

    def track(self, armors, t):
        """armors: list of solved dicts (name,type,color,xyz_in_world,ypr_in_world,
        ypd_in_world, center_px). Returns [target] or []."""
        if self.last_t is not None:
            dt = t - self.last_t
            if self.state != "lost" and dt > 0.1:
                self.state = "lost"
        self.last_t = t
        # Color filter: the game can misclassify the outpost's plate color at
        # range (red outpost detected 'blue'), so when enemy_color == 'any' we
        # track by NAME only — safe here because the turret is already driven
        # onto the specific outpost and we sort by image-center.
        if self.p.get("enemy_color") != "any":
            armors = [a for a in armors if a["color"] == self.p["enemy_color"]]
        # prefer image-center, then priority (all outpost = same priority here)
        armors.sort(key=lambda a: (a["center_px"][0] - 1440 / 2) ** 2
                    + (a["center_px"][1] - 1080 / 2) ** 2)

        if self.state == "lost":
            found = self._set_target(armors, t)
        else:
            found = self._update_target(armors, t)
        self._state_machine(found)

        if self.state != "lost" and self.target.diverged():
            self.state = "lost"
            return []
        nf = self.target.ekf.recent_nis_failures if self.target else [0]
        if self.target and sum(nf) >= 0.4 * self.target.ekf.window_size:
            self.state = "lost"
            return []
        if self.state == "lost":
            return []
        return [self.target]

    def _set_target(self, armors, t):
        if not armors:
            return False
        a = armors[0]
        if a["name"] == "outpost":
            P0 = [1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0]   # tracker.cpp:249
            self.target = Target(a, t, 0.2765, 3, P0)
        elif a["name"] == "base":
            P0 = [1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0]
            self.target = Target(a, t, 0.3205, 3, P0)
        else:
            P0 = [1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1]
            self.target = Target(a, t, 0.2, 4, P0)
        # propagate range-tunable omega-clamp params onto the fresh target
        self.target.p_omega_clamp_thr = self.p.get("omega_clamp_thr", 2.0)
        self.target.p_omega_lock = self.p.get("omega_lock", 2.513)
        self.target.p_omega_early = bool(self.p.get("omega_early", False))
        return True

    def _update_target(self, armors, t):
        self.target.predict_to(t)
        matched = [a for a in armors if a["name"] == self.target.name
                   and a["type"] == self.target.armor_type]
        if not matched:
            return False
        for a in matched:
            self.target.update(a)
        return True

    def _state_machine(self, found):
        s = self.state
        if s == "lost":
            if found:
                self.state = "detecting"; self.detect_count = 1
        elif s == "detecting":
            if found:
                self.detect_count += 1
                if self.detect_count >= self.p["min_detect_count"]:
                    self.state = "tracking"
            else:
                self.detect_count = 0; self.state = "lost"
        elif s == "tracking":
            if not found:
                self.temp_lost_count = 1; self.state = "temp_lost"
        elif s == "temp_lost":
            if found:
                self.state = "tracking"
            else:
                self.temp_lost_count += 1
                cap = (self.p["outpost_max_temp_lost_count"]
                       if self.target.name == "outpost"
                       else self.p["max_temp_lost_count"])
                if self.temp_lost_count > cap:
                    self.state = "lost"


class Aimer:
    """Flight-time iteration + coming/leaving plate selection (aimer.cpp)."""

    def __init__(self, params):
        self.p = params
        self.lock_id = -1
        self.debug_valid = False

    def aim(self, targets, t, bullet_speed):
        """Returns (control, yaw_deg, pitch_deg) in UE world (pitch up-positive)."""
        self.debug_valid = False
        if not targets:
            return (False, 0.0, 0.0)
        target = targets[0]
        g = self.p["g"]
        delay = (self.p["high_speed_delay_time"]
                 if target.ekf_x()[7] > self.p["decision_speed"]
                 else self.p["low_speed_delay_time"])
        if bullet_speed < 14:
            bullet_speed = 23.0
        future = t + self.p["detector_dt"] + delay
        target.predict_to(future)

        aim0 = self._choose_aim_point(target)
        if aim0 is None:
            return (False, 0.0, 0.0)
        xyz0 = aim0[:3]
        d0 = math.hypot(xyz0[0], xyz0[1])
        tr = trajectory(bullet_speed, d0, xyz0[2], g)
        if tr is None:
            return (False, 0.0, 0.0)

        prev_ft = tr[1]
        cur_traj = tr
        final_xyz = xyz0
        # copies for flight-time iteration (aimer.cpp:82-116)
        import copy
        for _ in range(10):
            it = copy.deepcopy(target)
            it.predict_to(future + prev_ft)
            aim = self._choose_aim_point(it)
            if aim is None:
                return (False, 0.0, 0.0)
            xyz = aim[:3]
            final_xyz = xyz
            d = math.hypot(xyz[0], xyz[1])
            cur_traj = trajectory(bullet_speed, d, xyz[2], g)
            if cur_traj is None:
                return (False, 0.0, 0.0)
            if abs(cur_traj[1] - prev_ft) < 0.001:
                break
            prev_ft = cur_traj[1]

        self.debug_valid = True
        yaw = math.atan2(final_xyz[1], final_xyz[0]) + math.radians(self.p["yaw_offset_deg"])
        pitch_world_neg = -(cur_traj[0] + math.radians(self.p["pitch_offset_deg"]))
        # aimer returns world-up-NEGATIVE; UE turret pitch is up-POSITIVE -> negate
        return (True, math.degrees(yaw), -math.degrees(pitch_world_neg))

    def _choose_aim_point(self, target):
        x = target.ekf_x()
        plates = target.armor_xyza_list()
        n = len(plates)
        if not target.jumped:
            return plates[0]
        center_yaw = math.atan2(x[2], x[0])
        delta = [limit_rad(plates[i][3] - center_yaw) for i in range(n)]

        # non-spin branch (aimer.cpp:163) — NOTE x[8] index is upstream's (radius);
        # kept faithfully, but outpost is forced past it by name check
        if abs(x[8]) <= 2 and target.name != "outpost":
            ids = [i for i in range(n) if abs(delta[i]) <= 60 / 57.3]
            if not ids:
                return None
            if len(ids) > 1:
                i0, i1 = ids[0], ids[1]
                if self.lock_id not in (i0, i1):
                    self.lock_id = i0 if abs(delta[i0]) < abs(delta[i1]) else i1
                return plates[self.lock_id]
            self.lock_id = -1
            return plates[ids[0]]

        if target.name == "outpost":
            coming, leaving = 70 / 57.3, 30 / 57.3
        else:
            coming = math.radians(self.p["coming_angle_deg"])
            leaving = math.radians(self.p["leaving_angle_deg"])
        for i in range(n):
            if abs(delta[i]) > coming:
                continue
            if x[7] > 0 and delta[i] < leaving:
                return plates[i]
            if x[7] < 0 and delta[i] > -leaving:
                return plates[i]
        return None


class Shooter:
    """Settle gate (shooter.cpp:19-41)."""

    def __init__(self, params):
        self.p = params
        self.last_yaw = None

    def shoot(self, control, yaw_deg, targets, aimer, gimbal_yaw_deg):
        if not control or not targets:
            self.last_yaw = yaw_deg
            return False
        tx, ty = targets[0].ekf_x()[0], targets[0].ekf_x()[2]
        tol = (self.p["second_tolerance_deg"] if math.hypot(tx, ty) > self.p["judge_distance"]
               else self.p["first_tolerance_deg"])
        fire = False
        if self.last_yaw is not None:
            d_cmd = abs((yaw_deg - self.last_yaw + 180) % 360 - 180)
            d_gim = abs((gimbal_yaw_deg - self.last_yaw + 180) % 360 - 180)
            if d_cmd < tol * 2 and d_gim < tol and aimer.debug_valid:
                fire = True
        self.last_yaw = yaw_deg
        return fire


class SpvisionAimPipeline:
    """detector-output -> PnP -> tracker -> aimer -> shooter, one step per frame."""

    def __init__(self, K, params=None):
        self.K = K
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.tracker = Tracker(self.p)
        self.aimer = Aimer(self.p)
        self.shooter = Shooter(self.p)

    def set_params(self, params):
        if params:
            self.p.update(params)

    def status(self):
        tgt = self.tracker.target
        base = {"n_solved": getattr(self, "last_n_solved", None),
                "n_raw": getattr(self, "last_n_raw", None),
                "colors": getattr(self, "last_colors", None)}
        if tgt is None:
            return {"state": self.tracker.state, "converged": False, **base}
        x = tgt.ekf_x()
        return {
            "state": self.tracker.state,
            "converged": bool(tgt.convergened()),
            "omega": round(float(x[7]), 3),
            "radius": round(float(x[8]), 4),
            "center": [round(float(x[0]), 2), round(float(x[2]), 2), round(float(x[4]), 2)],
            "update_count": tgt.update_count,
            **base,
        }

    def step(self, plates_det, turret_yaw_deg, turret_pitch_deg, t, bullet_speed):
        """plates_det: yolo detections (name,color,type via 'name' being 'outpost',
        keypoints, box). Solve each to world pose, feed tracker/aimer/shooter."""
        solved = []
        self.last_n_raw = len(plates_det)
        self.last_colors = [d["color"] for d in plates_det]
        for d in plates_det:
            pose = aim_solver.solve_plate_pose(d["keypoints"], self.K,
                                               turret_yaw_deg, turret_pitch_deg,
                                               armor_name=d["name"])
            if pose is None:
                continue
            solved.append({
                "name": d["name"],
                "type": "big" if d["name"] == "base" else "small",
                "color": d["color"],
                "xyz_in_world": pose["xyz"],
                "ypr_in_world": [pose["yaw"], 0.0, 0.0],
                "ypd_in_world": pose["ypd"],
                "center_px": (d["box"][0] + d["box"][2] / 2, d["box"][1] + d["box"][3] / 2),
            })
        self.last_n_solved = len(solved)
        targets = self.tracker.track(solved, t)
        control, yaw_deg, pitch_deg = self.aimer.aim(targets, t, bullet_speed)
        fire = self.shooter.shoot(control, yaw_deg, targets, self.aimer, turret_yaw_deg)
        return {"control": control, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
                "fire": fire, "state": self.tracker.state}


# ------------------------------------------------------------------------------
# Self-test: synthesize a rotating outpost, feed WORLD poses straight into the
# Tracker (bypass PnP), assert EKF converges to the true omega/radius and the
# aimer picks an incoming plate.
# ------------------------------------------------------------------------------
def _selftest():
    p = dict(DEFAULT_PARAMS)
    tr = Tracker(p)
    aimer = Aimer(p)
    true_center = np.array([4.0, 0.5, 0.0])   # 4m ahead, world x-fwd
    true_r = 0.2765
    true_omega = 2.513
    n = 3
    fps = 60.0
    a = 0.0
    got = []
    for k in range(int(3.0 * fps)):
        t = k / fps
        a = limit_rad(a + true_omega / fps)
        # the plate nearest to facing the sensor (max cos to -LOS) is "visible"
        best, best_face = None, -9
        for i in range(n):
            ang = limit_rad(a + i * 2 * PI / n)
            px = true_center[0] - true_r * math.cos(ang)
            py = true_center[1] - true_r * math.sin(ang)
            pz = true_center[2]
            # plate outward normal azimuth = ang (points away from center)
            los = math.atan2(py, px)
            face = math.cos(limit_rad(ang - los))   # ~1 when facing sensor
            if face > best_face:
                best_face, best = face, (px, py, pz, ang)
        if best_face < 0.3:   # no plate facing us this frame
            continue
        px, py, pz, ang = best
        armor = {
            "name": "outpost", "type": "small", "color": "red",
            "xyz_in_world": (px, py, pz),
            "ypr_in_world": [ang, 0.0, 0.0],
            "ypd_in_world": xyz2ypd((px, py, pz)),
            "center_px": (1440 / 2, 1080 / 2),
        }
        targets = tr.track([armor], t)
        if targets and targets[0].convergened():
            x = targets[0].ekf_x()
            got.append((abs(x[7]), x[8]))

    ok = False
    if got:
        est_omega = np.median([g[0] for g in got])
        est_r = np.median([g[1] for g in got])
        omega_err = abs(est_omega - true_omega) / true_omega
        r_ok = 0.05 < est_r < 0.5
        print(f"[selftest] converged {len(got)} frames | est |omega|={est_omega:.3f} "
              f"(true {true_omega}, err {omega_err*100:.1f}%) | est r={est_r:.3f} "
              f"(true {true_r})")
        ok = omega_err < 0.15 and r_ok
    else:
        print("[selftest] EKF never converged")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
