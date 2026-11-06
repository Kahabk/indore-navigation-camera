import argparse
import time
import cv2
import numpy as np

import config
from perception import FloorSegmenter
from planner import plan
from controller import Controller


class StuckDetector:
    """Driving forward but the image barely changes = stuck."""

    def __init__(self):
        self.t0 = None
        self.ref = None

    def update(self, frame, driving_forward):
        if not driving_forward:
            self.t0 = None
            self.ref = None
            return False
        now = time.time()
        g = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY)
        if self.t0 is None:
            self.t0, self.ref = now, g
            return False
        if now - self.t0 >= config.STUCK_TIME:
            diff = float(np.mean(cv2.absdiff(g, self.ref)))
            self.t0 = None
            self.ref = None
            return diff < config.STUCK_DIFF
        return False


def draw(frame, p, danger, state, drive, left, right, fps):
    out = frame.copy()
    layer = np.zeros_like(out)
    layer[p["safe"] > 0] = (0, 255, 0)
    layer[danger > 0] = (0, 0, 255)
    out = cv2.addWeighted(out, 0.7, layer, 0.3, 0)
    h, w = out.shape[:2]
    if p["target_x"] is not None:
        cv2.line(out, (w // 2, h - 5), (int(p["target_x"]), h // 2), (255, 0, 0), 2)
    status = "DRIVE ON" if drive else "DRIVE OFF"
    cv2.putText(out, f"{status} {state} L={left:.0f} R={right:.0f} {fps:.0f}fps",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(out, f"clear={p['clearance']:.2f} blocked={p['blocked']}",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true", help="start with drive ON")
    ap.add_argument("--no-robot", action="store_true", help="perception test only")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    seg = FloorSegmenter()
    ctrl = Controller()
    stuck_det = StuckDetector()
    bot = None
    if not args.no_robot:
        from esp32_client import Esp32Client
        bot = Esp32Client()

    margin = config.DANGER_MARGIN
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))

    drive = args.drive
    fps, t_prev = 0.0, time.time()
    last_state = None
    print("Click the CAMERA window. Keys: g = drive ON / reset, s = STOP, q = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -> stopping")
                if bot:
                    bot.stop()
                break
            frame = cv2.resize(frame, (config.FRAME_W, config.FRAME_H))

            floor, danger, danger_close = seg.segment(frame)
            safe_in = cv2.bitwise_and(floor, cv2.bitwise_not(cv2.dilate(danger, dil_kernel)))
            p = plan(safe_in)

            dark = float(frame.mean()) < config.DARK_LEVEL
            stuck = stuck_det.update(frame, drive and ctrl.state == "FORWARD"
                                     and getattr(ctrl, "_last_cmd", (0, 0)) == (1.0, 1.0))
            left, right = ctrl.command(p, danger_close or dark, stuck)
            ctrl._last_cmd = (left, right)

            if ctrl.state != last_state:
                print(f"[state] {ctrl.state}" + ("  (too dark)" if dark else ""))
                last_state = ctrl.state

            if bot:
                if drive:
                    bot.send(left, right)
                else:
                    bot.stop()

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

            cv2.imshow("Camera", draw(frame, p, danger, ctrl.state, drive,
                                      left if drive else 0, right if drive else 0, fps))
            cv2.imshow("Floor mask", safe_in)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                drive = True
                ctrl.reset()
            elif key == ord("s"):
                drive = False
                if bot:
                    bot.stop()
    finally:
        if bot:
            bot.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
