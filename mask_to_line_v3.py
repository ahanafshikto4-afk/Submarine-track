# -*- coding: utf-8 -*-

from pathlib import Path
from collections import deque
import csv
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = r"C:\work\yolo v8.4.24\v8\ultralytics-8.2.30\YellowPipe_Project\pipe_seg_v8s_v1\weights\v8s-seg-best.pt"
SOURCE_DIR = r"C:\codex\yellow_pipe_yolov8_seg_20260503\real_test_images"
OUT_DIR = r"C:\codex\yellow_pipe_yolov8_seg_20260503\line_output_v3"

FULL_CONF = 0.12
TILE_CONF = 0.10
MASK_THRESH = 0.25
FULL_IMGSZ = 1280
TILE_SIZE = 768
TILE_STRIDE = 512
TILE_IMGSZ = 768

MIN_MASK_AREA = 700
MIN_COLOR_AREA = 1200
MIN_LINE_POINTS = 40
SMOOTH_WINDOW = 31


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    path = str(path)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def masks_to_union(result, shape):
    h, w = shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return union

    masks = result.masks.data.cpu().numpy()
    for m in masks:
        mask = (m > MASK_THRESH).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        union = cv2.bitwise_or(union, mask)

    return union


def predict_full_mask(model, img):
    result = model.predict(
        img,
        imgsz=FULL_IMGSZ,
        conf=FULL_CONF,
        retina_masks=True,
        verbose=False,
        max_det=50,
    )[0]
    return masks_to_union(result, img.shape)


def predict_tiled_mask(model, img):
    h, w = img.shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)

    xs = tile_starts(w, TILE_SIZE, TILE_STRIDE)
    ys = tile_starts(h, TILE_SIZE, TILE_STRIDE)

    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + TILE_SIZE, w)
            y2 = min(y1 + TILE_SIZE, h)
            tile = img[y1:y2, x1:x2]
            result = model.predict(
                tile,
                imgsz=TILE_IMGSZ,
                conf=TILE_CONF,
                retina_masks=True,
                verbose=False,
                max_det=30,
            )[0]
            tile_mask = masks_to_union(result, tile.shape)
            union[y1:y2, x1:x2] = cv2.bitwise_or(union[y1:y2, x1:x2], tile_mask)

    return union


def fill_components(mask, min_area=MIN_MASK_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def cleanup_binary(mask, min_area=MIN_MASK_AREA, close_size=9, open_size=3):
    if mask is None or mask.size == 0:
        return mask

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return fill_components(mask, min_area=min_area)


def yellow_candidate_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)

    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)

    hsv_yellow = (h >= 12) & (h <= 62) & (s >= 35) & (v >= 65)
    channel_yellow = (
        (r16 >= 85)
        & (g16 >= 90)
        & (b16 <= 190)
        & ((g16 - b16) >= 18)
        & ((r16 - b16) >= 5)
    )
    bright_yellow = (
        (r16 >= 115)
        & (g16 >= 125)
        & (b16 <= 175)
        & ((r16 + g16) >= (2 * b16 + 45))
    )

    mask = (hsv_yellow | channel_yellow | bright_yellow).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    return cleanup_binary(mask, min_area=MIN_COLOR_AREA, close_size=11, open_size=3)


def keep_color_components_near_yolo(color_mask, yolo_mask):
    if cv2.countNonZero(color_mask) == 0:
        return color_mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats((color_mask > 0).astype(np.uint8), 8)
    kept = np.zeros_like(color_mask)

    if cv2.countNonZero(yolo_mask) == 0:
        if num <= 1:
            return kept
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept[labels == largest] = 255
        return kept

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    yolo_near = cv2.dilate(yolo_mask, kernel, iterations=1)

    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_COLOR_AREA:
            continue

        comp = labels == label
        overlap = int(np.count_nonzero(comp & (yolo_near > 0)))
        if overlap >= max(40, int(area * 0.005)):
            kept[comp] = 255

    if cv2.countNonZero(kept) == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept[labels == largest] = 255

    return kept


def build_final_mask(yolo_mask, color_mask):
    yolo_mask = cleanup_binary(yolo_mask, min_area=MIN_MASK_AREA, close_size=9, open_size=3)
    color_kept = keep_color_components_near_yolo(color_mask, yolo_mask)

    combined = cv2.bitwise_or(yolo_mask, color_kept)
    combined = cleanup_binary(combined, min_area=MIN_MASK_AREA, close_size=15, open_size=3)

    return combined, color_kept


def skeletonize_mask(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def skeleton_points_to_longest_path(points):
    if len(points) < 2:
        return None

    point_set = set(points)
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    def neighbors(p):
        x, y = p
        for dx, dy in directions:
            q = (x + dx, y + dy)
            if q in point_set:
                yield q

    def bfs(start):
        q = deque([start])
        parent = {start: None}
        dist = {start: 0}

        while q:
            p = q.popleft()
            for n in neighbors(p):
                if n not in parent:
                    parent[n] = p
                    dist[n] = dist[p] + 1
                    q.append(n)

        farthest = max(dist, key=dist.get)
        return farthest, parent, dist

    endpoints = []
    for p in point_set:
        degree = sum(1 for _ in neighbors(p))
        if degree == 1:
            endpoints.append(p)

    start = endpoints[0] if endpoints else next(iter(point_set))
    p1, _, _ = bfs(start)
    p2, parent, _ = bfs(p1)

    path = []
    cur = p2
    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    if len(path) < 2:
        return None

    return np.array(path, dtype=np.float32)


def split_skeleton_components(skeleton):
    binary = (skeleton > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []

    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_LINE_POINTS:
            continue
        ys, xs = np.where(labels == label)
        points = list(zip(xs.tolist(), ys.tolist()))
        components.append(points)

    return components


def smooth_path(path, window=SMOOTH_WINDOW):
    if path is None or len(path) < 5:
        return path

    window = int(window)
    if window % 2 == 0:
        window += 1

    if len(path) <= window:
        window = max(3, len(path) // 2 * 2 - 1)

    if window < 3:
        return path.astype(np.int32)

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    x = path[:, 0]
    y = path[:, 1]

    x_pad = np.pad(x, (pad, pad), mode="edge")
    y_pad = np.pad(y, (pad, pad), mode="edge")

    x_smooth = np.convolve(x_pad, kernel, mode="valid")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")

    smoothed = np.column_stack([x_smooth, y_smooth])
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed.astype(np.int32)


def tangent(line, at_start, n=18):
    if line is None or len(line) < 2:
        return None

    n = min(n, len(line) - 1)
    if at_start:
        v = line[0].astype(np.float32) - line[n].astype(np.float32)
    else:
        v = line[-1].astype(np.float32) - line[-1 - n].astype(np.float32)

    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None
    return v / norm


def angle_between(v1, v2):
    if v1 is None or v2 is None:
        return 180.0
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def support_ratio_between(p1, p2, support_mask, radius=10):
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1.0:
        return 1.0

    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    support = cv2.dilate((support_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    samples = max(8, int(dist / 4))
    good = 0
    h, w = support.shape[:2]

    for t in np.linspace(0.0, 1.0, samples):
        p = p1 * (1.0 - t) + p2 * t
        x = int(round(p[0]))
        y = int(round(p[1]))
        if 0 <= x < w and 0 <= y < h and support[y, x] > 0:
            good += 1

    return good / samples


def oriented_line(line, endpoint):
    if endpoint == "end":
        return line
    return line[::-1]


def bridge_lines(line_a, line_b, endpoint_a, endpoint_b):
    a = line_a if endpoint_a == "end" else line_a[::-1]
    b = line_b if endpoint_b == "start" else line_b[::-1]

    p1 = a[-1]
    p2 = b[0]
    bridge_len = max(2, int(np.linalg.norm(p2.astype(np.float32) - p1.astype(np.float32))))
    xs = np.linspace(p1[0], p2[0], bridge_len).astype(np.int32)
    ys = np.linspace(p1[1], p2[1], bridge_len).astype(np.int32)
    bridge = np.column_stack([xs, ys])
    return np.vstack([a, bridge, b]).astype(np.int32)


def merge_lines_safely(lines, support_mask, max_dist=120, max_angle=40, min_support=0.68):
    lines = [line for line in lines if line is not None and len(line) >= MIN_LINE_POINTS]
    changed = True

    while changed and len(lines) > 1:
        changed = False
        best = None

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a = lines[i]
                b = lines[j]

                b_start_out = tangent(b, True)
                b_end_out = tangent(b, False)
                candidates = [
                    ("end", "start", a[-1], b[0], tangent(a, False), -b_start_out if b_start_out is not None else None),
                    ("end", "end", a[-1], b[-1], tangent(a, False), -b_end_out if b_end_out is not None else None),
                    ("start", "start", a[0], b[0], tangent(a, True), -b_start_out if b_start_out is not None else None),
                    ("start", "end", a[0], b[-1], tangent(a, True), -b_end_out if b_end_out is not None else None),
                ]

                for end_a, end_b, p1, p2, v1, v2 in candidates:
                    delta = p2.astype(np.float32) - p1.astype(np.float32)
                    dist = float(np.linalg.norm(delta))
                    if dist > max_dist or dist < 1.0:
                        continue

                    gap_dir = delta / dist
                    if v1 is None or v2 is None:
                        continue

                    if float(np.dot(v1, gap_dir)) < 0.45:
                        continue
                    if float(np.dot(v2, gap_dir)) < 0.45:
                        continue

                    angle = angle_between(v1, v2)
                    if angle > max_angle:
                        continue

                    support = support_ratio_between(p1, p2, support_mask, radius=12)
                    if support < min_support:
                        continue

                    score = dist + angle * 2.0 + (1.0 - support) * 80.0
                    if best is None or score < best[0]:
                        best = (score, i, j, end_a, end_b)

        if best is not None:
            _, i, j, end_a, end_b = best
            merged = bridge_lines(lines[i], lines[j], end_a, end_b)
            merged = smooth_path(merged, SMOOTH_WINDOW)
            lines = [line for k, line in enumerate(lines) if k not in (i, j)]
            lines.append(merged)
            changed = True

    return lines


def centerlines_from_mask(mask, support_mask):
    skeleton = skeletonize_mask(mask)
    paths = []

    for points in split_skeleton_components(skeleton):
        raw_path = skeleton_points_to_longest_path(points)
        if raw_path is None or len(raw_path) < MIN_LINE_POINTS:
            continue
        paths.append(smooth_path(raw_path, SMOOTH_WINDOW))

    paths = merge_lines_safely(paths, support_mask)
    return skeleton, paths


def draw_result(image, final_mask, yolo_mask, color_mask, skeleton, centerlines):
    out = image.copy()

    color_layer = np.zeros_like(out)
    color_layer[:, :, 1] = final_mask
    out = cv2.addWeighted(out, 1.0, color_layer, 0.33, 0)

    yolo_edges = cv2.Canny(yolo_mask, 50, 150)
    out[yolo_edges > 0] = (0, 255, 255)

    color_edges = cv2.Canny(color_mask, 50, 150)
    out[color_edges > 0] = (255, 255, 0)

    ys, xs = np.where(skeleton > 0)
    out[ys, xs] = (255, 0, 0)

    for line in centerlines:
        if line is not None and len(line) > 1:
            cv2.polylines(out, [line], isClosed=False, color=(0, 0, 255), thickness=4)

    return out


def path_length(line):
    if line is None or len(line) < 2:
        return 0.0
    diffs = np.diff(line.astype(np.float32), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def write_csv(path, centerlines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipe_id", "x", "y"])
        for line_id, centerline in enumerate(centerlines, start=1):
            for x, y in centerline:
                writer.writerow([line_id, int(x), int(y)])


def process_one_image(model, image_path, out_dir):
    img = imread_unicode(image_path)
    if img is None:
        print(f"Read failed: {image_path}")
        return

    full_mask = predict_full_mask(model, img)
    tile_mask = predict_tiled_mask(model, img)
    yolo_mask = cv2.bitwise_or(full_mask, tile_mask)
    color_mask = yellow_candidate_mask(img)
    final_mask, color_kept = build_final_mask(yolo_mask, color_mask)
    support_mask = cv2.bitwise_or(final_mask, color_kept)

    if cv2.countNonZero(final_mask) == 0:
        print(f"No valid mask: {image_path.name}")
        return

    skeleton, centerlines = centerlines_from_mask(final_mask, support_mask)
    if not centerlines:
        print(f"No valid centerline: {image_path.name}")
        return

    stem = image_path.stem
    result_img = draw_result(img, final_mask, yolo_mask, color_kept, skeleton, centerlines)

    imwrite_unicode(out_dir / f"{stem}_line.png", result_img)
    imwrite_unicode(out_dir / f"{stem}_mask.png", final_mask)
    imwrite_unicode(out_dir / f"{stem}_yolo_mask.png", yolo_mask)
    imwrite_unicode(out_dir / f"{stem}_color_mask.png", color_kept)
    imwrite_unicode(out_dir / f"{stem}_skeleton.png", skeleton)
    write_csv(out_dir / f"{stem}_line_points.csv", centerlines)

    line_lengths = [round(path_length(line), 1) for line in centerlines]
    print(
        f"Done: {image_path.name}, lines={len(centerlines)}, "
        f"yolo_area={cv2.countNonZero(yolo_mask)}, "
        f"color_area={cv2.countNonZero(color_kept)}, "
        f"final_area={cv2.countNonZero(final_mask)}, "
        f"lengths={line_lengths}"
    )


def main():
    source_dir = Path(SOURCE_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(source_dir.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in: {source_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    for image_path in image_paths:
        process_one_image(model, image_path, out_dir)

    print(f"Finished. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
