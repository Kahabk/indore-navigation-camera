import cv2
import numpy as np

import config


def plan(floor_mask):
    """Find the best connected free-floor corridor in the image."""
    H, W = floor_mask.shape

    # 1. Expand obstacles = shrink floor (clearance margin)
    k = config.CLEARANCE_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    safe = cv2.erode(floor_mask, kernel)

    # 2. Keep only the floor region connected to the robot (bottom-center)
    _, lab = cv2.connectedComponents(safe)
    band = lab[H - 12:H, W // 3: 2 * W // 3]
    vals = band[band > 0]
    if vals.size == 0:
        connected = np.zeros((H, W), bool)
    else:
        comp = np.bincount(vals).argmax()
        connected = lab == comp

    # 3. Free height per column, counted upward from the bottom edge
    rev = ~connected[::-1, :]
    has_block = rev.any(axis=0)
    first_block = rev.argmax(axis=0)
    heights = np.where(has_block, first_block, H).astype(np.float32)

    pad = config.SMOOTH_COLS // 2
    padded = np.pad(heights, pad, mode="edge")
    heights = np.convolve(padded, np.ones(config.SMOOTH_COLS) / config.SMOOTH_COLS, mode="valid")

    # 4. Find runs of usable columns
    ok = heights >= config.MIN_FREE_RATIO * H
    runs, start = [], None
    for x in range(W + 1):
        val = ok[x] if x < W else False
        if val and start is None:
            start = x
        elif not val and start is not None:
            runs.append((start, x - 1))
            start = None

    best, best_score = None, -1e9
    for s, e in runs:
        width = e - s + 1
        if width < config.MIN_CORRIDOR_RATIO * W:
            continue
        depth = heights[s:e + 1].mean() / H
        center = (s + e) / 2
        score = depth + 0.5 * width / W - 0.4 * abs(center - W / 2) / (W / 2)
        if score > best_score:
            deepest = s + int(np.argmax(heights[s:e + 1]))
            best_score = score
            best = 0.5 * center + 0.5 * deepest

    c0, c1 = W // 3, 2 * W // 3
    clearance = float(heights[c0:c1].mean() / H)
    blocked = best is None or clearance < config.STOP_RATIO

    return {
        "target_x": None if best is None else float(best),
        "clearance": clearance,
        "blocked": blocked,
        "left_free": float(heights[: W // 2].mean() / H),
        "right_free": float(heights[W // 2:].mean() / H),
        "safe": connected.astype(np.uint8) * 255,
        "width": W,
    }
