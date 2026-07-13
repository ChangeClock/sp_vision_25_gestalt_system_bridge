#!/usr/bin/env python3
"""Offline armor-detection harness replicating tasks/auto_aim/yolos/yolo11.cpp.

Purpose: V1 recognition-rate gate for the gestalt_system bridge — run the
sp_vision yolo11 OpenVINO model on (a) the repo's demo.avi as a correctness
baseline, then (b) UE-rendered game frames to measure the render-domain gap.

Preprocess (mirrors YOLO11::detect): BGR frame -> top-left letterbox into
640x640 (scale = min(640/h, 640/w), zero pad) -> RGB f32 /255 NCHW.
Postprocess (mirrors YOLO11::parse): output -> transpose -> rows of
[xywh(4) | class scores(38) | keypoints(8)] -> score>=0.7 -> NMS(0.3)
-> class table decode.

Usage:
  python tools/vino_detect.py --video assets/demo/demo.avi --save-annot 20
  python tools/vino_detect.py --frames <dir-of-jpg> --out results.jsonl
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import openvino as ov

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = ["blue", "red", "extinguish", "purple"]
NAMES = ["one", "two", "three", "four", "five", "sentry", "outpost", "base", "not_armor"]
# armor.hpp armor_properties — index = class_id -> (color, name, type)
ARMOR_PROPERTIES = [
    ("blue", "sentry", "small"), ("red", "sentry", "small"), ("extinguish", "sentry", "small"),
    ("blue", "one", "small"), ("red", "one", "small"), ("extinguish", "one", "small"),
    ("blue", "two", "small"), ("red", "two", "small"), ("extinguish", "two", "small"),
    ("blue", "three", "small"), ("red", "three", "small"), ("extinguish", "three", "small"),
    ("blue", "four", "small"), ("red", "four", "small"), ("extinguish", "four", "small"),
    ("blue", "five", "small"), ("red", "five", "small"), ("extinguish", "five", "small"),
    ("blue", "outpost", "small"), ("red", "outpost", "small"), ("extinguish", "outpost", "small"),
    ("blue", "base", "big"), ("red", "base", "big"), ("extinguish", "base", "big"), ("purple", "base", "big"),
    ("blue", "base", "small"), ("red", "base", "small"), ("extinguish", "base", "small"), ("purple", "base", "small"),
    ("blue", "three", "big"), ("red", "three", "big"), ("extinguish", "three", "big"),
    ("blue", "four", "big"), ("red", "four", "big"), ("extinguish", "four", "big"),
    ("blue", "five", "big"), ("red", "five", "big"), ("extinguish", "five", "big"),
]
CLASS_NUM = 38
SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3


def preprocess(bgr):
    h, w = bgr.shape[:2]
    scale = min(640.0 / h, 640.0 / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None], scale


def parse(output, scale, score_threshold):
    out = np.squeeze(output)  # [4+38+8, N]
    out = out.T  # rows = candidates
    scores = out[:, 4:4 + CLASS_NUM]
    best_ids = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(out)), best_ids]
    keep = best_scores >= score_threshold
    if not keep.any():
        return []
    out, best_ids, best_scores = out[keep], best_ids[keep], best_scores[keep]
    boxes = []
    for row in out:
        x, y, w, h = row[:4]
        boxes.append([int((x - 0.5 * w) / scale), int((y - 0.5 * h) / scale),
                      int(w / scale), int(h / scale)])
    indices = cv2.dnn.NMSBoxes(boxes, best_scores.tolist(), score_threshold, NMS_THRESHOLD)
    dets = []
    for i in np.array(indices).flatten():
        kps = out[i, 4 + CLASS_NUM:4 + CLASS_NUM + 8] / scale
        # yolo11.cpp sort_keypoints: order TL,TR,BR,BL (top pair by y, then x).
        pts = kps.reshape(4, 2)
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2][np.argsort(pts[:2, 0])]
        bot = pts[2:][np.argsort(pts[2:, 0])]
        kps = np.array([top[0], top[1], bot[1], bot[0]]).reshape(8)
        color, name, atype = ARMOR_PROPERTIES[best_ids[i]]
        dets.append({
            "class_id": int(best_ids[i]),
            "confidence": float(best_scores[i]),
            "color": color, "name": name, "type": atype,
            "box": boxes[i],
            "keypoints": kps.reshape(4, 2).tolist(),
        })
    return dets


def annotate(img, dets):
    for d in dets:
        pts = np.array(d["keypoints"], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        x, y, w, h = d["box"]
        cv2.putText(img, f'{d["color"]}-{d["name"]} {d["confidence"]:.2f}',
                    (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def frame_iter(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"frame_{idx:06d}", frame
            idx += 1
        cap.release()
    else:
        files = sorted(f for f in os.listdir(args.frames)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        for f in files:
            img = cv2.imread(os.path.join(args.frames, f))
            if img is not None:
                yield os.path.splitext(f)[0], img


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames")
    ap.add_argument("--model", default=os.path.join(REPO, "assets", "yolo11.xml"))
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--out", default=None, help="per-frame JSONL output path")
    ap.add_argument("--save-annot", type=int, default=0,
                    help="save this many annotated frames (with >=1 detection)")
    ap.add_argument("--annot-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max frames to process")
    args = ap.parse_args()

    core = ov.Core()
    model = core.compile_model(args.model, args.device)

    out_path = args.out or os.path.join(REPO, "detect_results.jsonl")
    annot_dir = args.annot_dir or os.path.join(REPO, "detect_annot")
    if args.save_annot:
        os.makedirs(annot_dir, exist_ok=True)

    total = 0
    with_det = 0
    class_counts = {}
    conf_sum = 0.0
    det_count = 0
    saved = 0
    t0 = cv2.getTickCount()
    with open(out_path, "w", encoding="utf-8") as fp:
        for name, frame in frame_iter(args):
            tensor, scale = preprocess(frame)
            dets = parse(model(tensor)[0], scale, args.score_threshold)
            total += 1
            if dets:
                with_det += 1
                for d in dets:
                    key = f'{d["color"]}-{d["name"]}'
                    class_counts[key] = class_counts.get(key, 0) + 1
                    conf_sum += d["confidence"]
                    det_count += 1
                if args.save_annot and saved < args.save_annot:
                    cv2.imwrite(os.path.join(annot_dir, f"{name}.jpg"), annotate(frame.copy(), dets))
                    saved += 1
            fp.write(json.dumps({"frame": name, "detections": dets}) + "\n")
            if args.limit and total >= args.limit:
                break
    dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
    summary = {
        "frames": total,
        "frames_with_detection": with_det,
        "detection_frame_rate": round(with_det / total, 4) if total else 0,
        "detections": det_count,
        "mean_confidence": round(conf_sum / det_count, 4) if det_count else 0,
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
        "fps": round(total / dt, 1) if dt > 0 else 0,
        "model": os.path.basename(args.model),
        "score_threshold": args.score_threshold,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
