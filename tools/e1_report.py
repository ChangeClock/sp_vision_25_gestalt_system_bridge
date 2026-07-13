#!/usr/bin/env python3
"""E1 quantifier: detection rate on frames where an enemy is GEOMETRICALLY in view.

Joins the capture sidecar (per-frame turret pose + ground-truth robot snapshot)
with vino_detect per-frame results, labels each frame "enemy in FOV" from pure
geometry (bearing within horizontal FOV, distance within range; LOS/occlusion
NOT modeled — labels are optimistic, so the reported rate is a lower bound),
and reports detection rate overall and by distance band.

Usage:
  python tools/e1_report.py --sidecar <phase>.jsonl --dets <phase>.det.jsonl \
      --fov 45 [--max-range-cm 1500] [--enemy-color red]
"""

import argparse
import json
import math


def load_jsonl(path):
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--fov", type=float, required=True, help="horizontal FOV deg")
    ap.add_argument("--max-range-cm", type=float, default=1500.0)
    ap.add_argument("--min-range-cm", type=float, default=80.0)
    ap.add_argument("--enemy-color", default="red")
    ap.add_argument("--fov-margin-deg", type=float, default=3.0)
    args = ap.parse_args()

    dets_by_frame = {}
    for row in load_jsonl(args.dets):
        # vino_detect frame names: f000123 / frame_000123
        digits = "".join(ch for ch in row["frame"] if ch.isdigit())
        dets_by_frame[int(digits)] = row["detections"]

    bands = [(0, 400, "0-4m"), (400, 800, "4-8m"), (800, 1e9, ">8m")]
    stats = {name: {"vis": 0, "hit": 0} for _, _, name in bands}
    total_vis = 0
    total_hit = 0
    no_enemy_frames = 0
    no_enemy_det_frames = 0
    half_fov = args.fov / 2.0 - args.fov_margin_deg

    for row in load_jsonl(args.sidecar):
        idx = row["idx"]
        own = row.get("own") or {}
        yaw = row.get("yaw")
        if idx not in dets_by_frame or yaw is None or own.get("x") is None:
            continue
        own_team = own.get("team")
        nearest = None
        for r in row.get("robots") or []:
            if r.get("dead") or r.get("team") == own_team or r.get("pid") == own.get("pid"):
                continue
            if r.get("x") is None:
                continue
            dx, dy = r["x"] - own["x"], r["y"] - own["y"]
            dist = math.hypot(dx, dy)
            if dist < args.min_range_cm or dist > args.max_range_cm:
                continue
            bearing = math.degrees(math.atan2(dy, dx))
            diff = (bearing - yaw + 180.0) % 360.0 - 180.0
            if abs(diff) > half_fov:
                continue
            if nearest is None or dist < nearest:
                nearest = dist

        enemy_dets = [d for d in dets_by_frame[idx] if d["color"] == args.enemy_color]
        if nearest is None:
            no_enemy_frames += 1
            if enemy_dets:
                no_enemy_det_frames += 1
            continue
        total_vis += 1
        hit = 1 if enemy_dets else 0
        total_hit += hit
        for lo, hi, name in bands:
            if lo <= nearest < hi:
                stats[name]["vis"] += 1
                stats[name]["hit"] += hit
                break

    summary = {
        "frames_enemy_in_fov": total_vis,
        "frames_hit": total_hit,
        "detection_rate_visible": round(total_hit / total_vis, 4) if total_vis else None,
        "by_distance": {
            name: {
                "frames": s["vis"],
                "rate": round(s["hit"] / s["vis"], 4) if s["vis"] else None,
            }
            for name, s in stats.items()
        },
        "frames_no_enemy_in_fov": no_enemy_frames,
        "enemy_dets_on_no_enemy_frames": no_enemy_det_frames,
        "notes": "geometry-only labels (no LOS/occlusion) -> rate is a LOWER bound; "
                 "enemy_dets_on_no_enemy_frames mixes occluded-truth and false positives",
        "fov": args.fov,
        "max_range_cm": args.max_range_cm,
        "enemy_color": args.enemy_color,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
