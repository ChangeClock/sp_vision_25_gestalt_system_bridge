#!/usr/bin/env python3
"""sp_vision-faithful aim solving for the gestalt UE simulator bridge.

Ports the aiming math from tasks/auto_aim/solver.cpp + tools/trajectory.cpp +
tasks/auto_aim/aimer.cpp, adapted to the UE bridge's geometry:

  - The "camera" IS the vehicle's gun-view camera (frames come from window
    capture of the first-person view), so sp_vision's R_camera2gimbal /
    t_camera2gimbal (their robot's physical mount calibration, ~9.5cm offsets)
    do NOT apply. The camera pose in the UE world comes directly from the
    telemetry TurretYaw/TurretPitch (degrees, muzzle world rotation) — no IMU
    quaternion chain needed.
  - UE frames: world Z-up, yaw = FRotator.Yaw (deg, +X toward +Y), pitch
    up-positive. OpenCV camera frame: x-right, y-down, z-forward.
  - Ballistics: trajectory.cpp closed-form (no drag), g = 9.7833, low arc.
    Muzzle ≈ camera pivot (gun cam), so PnP-relative height feeds `h` directly.

Solve chain: 4 keypoints (px) -> solvePnP(IPPE) -> tvec in camera frame (m)
-> rotate into UE world by turret yaw/pitch -> horizontal distance d + height h
-> ballistic launch pitch -> absolute UE world (yaw_deg, pitch_deg) command.
"""

import math

import cv2
import numpy as np

# solver.cpp:12-14
LIGHTBAR_LENGTH = 56e-3
SMALL_ARMOR_WIDTH = 135e-3
BIG_ARMOR_WIDTH = 230e-3
# The GAME's projectile sim: Ballistic.csv Cal17mm fGravity=981 (sp_vision's own
# constant is 9.7833 — use the simulator's truth here).
GRAVITY = 9.81

# Ballistic.csv Cal17mm fSpeed=2450 cm/s minus ShooterSpeedSpreadPara (0..30):
# muzzle 24.2-24.5 m/s; ShooterRealSpeed attr 10000030 carries the per-shot value.
DEFAULT_BULLET_SPEED = 24.3

BIG_ARMOR_NAMES = {"base"}  # armor.hpp table: big = base/3/4/5-big variants; outpost is small

# Gestalt-simulator calibration (outpost benchmark, GT-anchored): the detector's
# keypoints on the RENDERED lightbars sit inside the physical 135x56 model —
# effective quad ~113x47mm (range solves 1.192x true with the stock model;
# median over 2704 sorted-keypoint solves at a fixed 4.28m standoff).
# IMPORTANT: planar 4-point PnP fits ANY width-only change exactly by tilting
# the plate (homography ambiguity) — range only responds to scaling BOTH
# dimensions, so calibration is a uniform quad scale, not a width override.
PER_NAME_QUAD_SCALE = {"outpost": 1.0 / 1.192}
DEFAULT_QUAD_SCALE = 1.0 / 1.192  # same render/detector effect presumably applies to robots

# Static aim bias (the sp_vision pitch_offset/yaw_offset calibration slot).
# GT pitch-sweep on the outpost benchmark: the vision-solved ballistic pitch
# already sits at the empirical optimum (+1.5 deg vs the coarse GT estimate,
# whose plate-z/muzzle-height assumptions carry that bias) — keep 0.
PITCH_OFFSET_DEG = 0.0
YAW_OFFSET_DEG = 0.0


def object_points(width: float, length: float = LIGHTBAR_LENGTH) -> np.ndarray:
    """solver.cpp:21-25 — armor frame x=outward normal, y=width(left+), z=up.
    Keypoint order TL, TR, BR, BL (armor.cpp:63-69)."""
    return np.array(
        [[0, width / 2, length / 2], [0, -width / 2, length / 2],
         [0, -width / 2, -length / 2], [0, width / 2, -length / 2]],
        np.float64,
    )


def camera_matrix(width_px: int, height_px: int, fov_h_deg: float, cy_px=None) -> np.ndarray:
    """Ideal pinhole for the UE render (no lens distortion). UE FOV is horizontal.

    cy_px: true principal-point row. When the captured frame is cropped at the
    BOTTOM (to drop the version footer), the optical axis still projects to the
    ORIGINAL full-frame center row, so cy must be full_H/2 = crop_H/2 + crop/2,
    NOT crop_H/2 — otherwise the principal point shifts up ~14px ≈ 0.44° pitch
    bias (spvision-alignment-plan §a.2). Pass cy_px = full_grab_height/2.
    """
    fx = (width_px / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
    cy = cy_px if cy_px is not None else height_px / 2.0
    return np.array([[fx, 0, width_px / 2.0], [0, fx, cy], [0, 0, 1]], np.float64)


def ue_cam_to_world(turret_yaw_deg: float, turret_pitch_deg: float) -> np.ndarray:
    """Rotation taking OpenCV-camera-frame vectors to UE world vectors.

    UE camera basis in world (FRotationMatrix, roll=0):
      forward F = (cp*cy, cp*sy, sp), right R = (-sy_?, ...) — derived from
      yaw about +Z (X toward Y) then pitch about the right axis:
      F = (cos p * cos y, cos p * sin y, sin p)
      Rt = (-sin y, cos y, 0)            # world right of the camera
      U  = F x Rt? — use U = Rt x F (UE left-handed: U = cross(Rt, F))
    OpenCV (x_c right, y_c down, z_c forward) -> world: x_c*Rt - y_c*U + z_c*F.
    """
    y = math.radians(turret_yaw_deg)
    p = math.radians(turret_pitch_deg)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    rt = np.array([-math.sin(y), math.cos(y), 0.0])
    up = np.cross(f, rt)  # (-sp*cy, -sp*sy, cp): identity pose -> (0,0,1) ✓
    # Columns map camera axes (right, down, forward) into world.
    return np.column_stack([rt, -up, f])


def solve_launch_pitch(v0: float, d: float, h: float):
    """trajectory.cpp:9-30 — closed-form low-arc launch pitch (rad, up+), or None."""
    a = GRAVITY * d * d / (2.0 * v0 * v0)
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


def solve_plate_pose(keypoints_px, k_matrix, turret_yaw_deg, turret_pitch_deg,
                     armor_name="outpost"):
    """PnP one armor plate to a GIMBAL-relative world pose for the sp_vision
    tracker. Returns {xyz (m, world x-fwd/y-left/z-up, origin=gimbal), yaw (plate
    facing/normal azimuth rad, ypr[0]), ypd (yaw,pitch,dist rad/rad/m)} or None.
    The plate's OUTWARD NORMAL is the armor-object +x axis (solver.cpp:21 object
    model), so plate world normal = R_cam2world @ R_armor2cam @ [1,0,0]."""
    width = BIG_ARMOR_WIDTH if armor_name in BIG_ARMOR_NAMES else SMALL_ARMOR_WIDTH
    scale = PER_NAME_QUAD_SCALE.get(armor_name, DEFAULT_QUAD_SCALE)
    obj = object_points(width * scale, LIGHTBAR_LENGTH * scale)
    kps = np.asarray(keypoints_px, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, kps, k_matrix, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k_matrix, np.zeros(5))
    if float(np.mean(np.linalg.norm(proj.reshape(4, 2) - kps, axis=1))) > 4.0:
        return None
    R_c2w = ue_cam_to_world(turret_yaw_deg, turret_pitch_deg)
    xyz = R_c2w @ tvec.reshape(3)
    R_armor2cam, _ = cv2.Rodrigues(rvec)
    normal_world = R_c2w @ (R_armor2cam @ np.array([1.0, 0.0, 0.0]))
    plate_yaw = math.atan2(normal_world[1], normal_world[0])
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    d = math.sqrt(x * x + y * y + z * z)
    if d < 0.3:
        return None
    return {
        "xyz": (x, y, z),
        "yaw": plate_yaw,
        "ypd": (math.atan2(y, x), math.atan2(z, math.hypot(x, y)), d),
    }


def solve_aim(keypoints_px, k_matrix, turret_yaw_deg, turret_pitch_deg,
              bullet_speed=DEFAULT_BULLET_SPEED, armor_name="outpost"):
    """Full sp_vision-style aim solve.

    Returns dict {yaw_deg, pitch_deg, distance_m, target_rel_ue (x,y,z m),
    fly_time_s} — yaw/pitch are ABSOLUTE UE world turret commands — or None.
    """
    width = BIG_ARMOR_WIDTH if armor_name in BIG_ARMOR_NAMES else SMALL_ARMOR_WIDTH
    scale = PER_NAME_QUAD_SCALE.get(armor_name, DEFAULT_QUAD_SCALE)
    obj = object_points(width * scale, LIGHTBAR_LENGTH * scale)
    kps = np.asarray(keypoints_px, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, kps, k_matrix, np.zeros(5),
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    # Quality gate: reprojection residual. Occluded/mangled quads (clipped by
    # terrain, motion blur) solve to garbage poses — a real pipeline rejects
    # them via its tracker; offline we use the residual directly.
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k_matrix, np.zeros(5))
    reproj_err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - kps, axis=1)))
    if reproj_err > 4.0:
        return None
    rel_ue = ue_cam_to_world(turret_yaw_deg, turret_pitch_deg) @ tvec.reshape(3)
    x, y, z = float(rel_ue[0]), float(rel_ue[1]), float(rel_ue[2])
    d = math.hypot(x, y)
    if d < 0.3:
        return None
    if bullet_speed < 14:
        bullet_speed = DEFAULT_BULLET_SPEED  # aimer.cpp:43 guard
    solved = solve_launch_pitch(bullet_speed, d, z)
    if solved is None:
        return None
    launch_pitch, fly_time = solved
    return {
        "yaw_deg": math.degrees(math.atan2(y, x)) + YAW_OFFSET_DEG,
        "pitch_deg": math.degrees(launch_pitch) + PITCH_OFFSET_DEG,  # UE up-positive
        "distance_m": d,
        "target_rel_ue": (x, y, z),
        "fly_time_s": fly_time,
        "reproj_err_px": reproj_err,
    }
