import time

import config


class Controller:
    """States: FORWARD, ARC, SPIN, SCAN, WAIT, RECOVER, LOST.
    Outputs only -1/0/+1 per wheel (the ESP32 applies your tuned PWM)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "FORWARD"
        self.err_s = 0.0
        self.turn_t0 = None
        self.fwd_t0 = None
        self.scan_dir = 1
        self.scan_t0 = 0.0
        self.scan_stop_until = 0.0
        self.scan_fail = 0
        self.last_scan_end = -1e9
        self.last_dir = 1
        self.clear_until = 0.0
        self.recover_until = 0.0
        self.recover_dir = 1
        self.lost = False

    def _spin(self, direction, on_time, now):
        if self.turn_t0 is None:
            self.turn_t0 = now
        phase = (now - self.turn_t0) % (on_time + config.TURN_OFF)
        if phase < on_time:
            return (1.0, -1.0) if direction > 0 else (-1.0, 1.0)
        return (0.0, 0.0)

    def _forward(self, clearance, now):
        if clearance >= config.SLOW_RATIO:
            self.fwd_t0 = None
            return (1.0, 1.0)
        if self.fwd_t0 is None:
            self.fwd_t0 = now
        phase = (now - self.fwd_t0) % (config.CREEP_ON + config.CREEP_OFF)
        return (1.0, 1.0) if phase < config.CREEP_ON else (0.0, 0.0)

    def command(self, p, danger_close=False, stuck=False):
        now = time.time()

        # 1. Person / pet / stairs close in front: stop and wait
        if danger_close:
            self.clear_until = now + config.PERSON_CLEAR_TIME
        if now < self.clear_until:
            self.state = "WAIT"
            self.turn_t0 = None
            return (0.0, 0.0)

        # 2. Gave up searching: stay stopped until reset (press g)
        if self.lost:
            self.state = "LOST"
            return (0.0, 0.0)

        # 3. Stuck: spin away from the last direction
        if stuck and self.state != "RECOVER":
            self.state = "RECOVER"
            self.recover_until = now + config.RECOVER_TIME
            self.recover_dir = -self.last_dir
            self.turn_t0 = None
        if self.state == "RECOVER":
            if now < self.recover_until:
                return self._spin(self.recover_dir, config.TURN_ON_MAX, now)
            self.state = "FORWARD"
            self.turn_t0 = None

        # 4. Blocked: stop, then scan for open space
        if p["blocked"] or (self.state == "SCAN" and p["clearance"] < config.RESUME_RATIO):
            if self.state != "SCAN":
                if now - self.last_scan_end < config.SCAN_REPEAT_WINDOW:
                    self.scan_fail += 1
                else:
                    self.scan_fail = 0
                pref = 1 if p["right_free"] >= p["left_free"] else -1
                if self.scan_fail >= 2:
                    pref = -self.scan_dir        # this side keeps failing, try the other
                    self.scan_fail = 0
                self.scan_dir = pref
                self.state = "SCAN"
                self.scan_t0 = now
                self.scan_stop_until = now + config.STOP_PAUSE
                self.turn_t0 = None
            if now - self.scan_t0 > config.SCAN_TIMEOUT:
                self.lost = True
                return (0.0, 0.0)
            if now < self.scan_stop_until:
                return (0.0, 0.0)
            self.last_dir = self.scan_dir
            return self._spin(self.scan_dir, config.TURN_ON_MAX, now)

        if self.state == "SCAN":
            self.last_scan_end = now
            self.state = "FORWARD"
            self.turn_t0 = None
            self.err_s = 0.0

        # 5. Normal driving: smoothed steering error with hysteresis
        half = p["width"] / 2
        err = (p["target_x"] - half) / half
        self.err_s = config.ERR_SMOOTH * self.err_s + (1 - config.ERR_SMOOTH) * err
        a = abs(self.err_s)
        thr = config.EXIT_ERR if self.state in ("ARC", "SPIN") else config.ENTER_ERR

        if a <= thr:
            self.state = "FORWARD"
            self.turn_t0 = None
            return self._forward(p["clearance"], now)

        direction = 1 if self.err_s > 0 else -1
        self.last_dir = direction
        if a > config.SPIN_ERR:
            self.state = "SPIN"
            on = config.TURN_ON_MIN + (config.TURN_ON_MAX - config.TURN_ON_MIN) * min(a, 1.0)
            return self._spin(direction, on, now)

        # moderate error: smooth arc while moving (one wheel only)
        self.state = "ARC"
        self.turn_t0 = None
        return (1.0, 0.0) if direction > 0 else (0.0, 1.0)
