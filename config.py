# ---------- Camera ----------
CAMERA_INDEX = 0            # /dev/video0
FRAME_W, FRAME_H = 320, 240

# ---------- Serial (ESP32 over USB) ----------
SERIAL_PORT = None          # None = auto-detect, or e.g. "/dev/ttyUSB0"
BAUD = 115200
INVERT_LEFT = False         # flip if a wheel spins the wrong way
INVERT_RIGHT = False

# ---------- Segmentation model ----------
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_INPUT_W, MODEL_INPUT_H = 320, 256   # must be multiples of 32
FLOOR_LABELS = {"floor", "rug"}
TEMPORAL_ALPHA = 0.5

# ---------- Planner ----------
CLEARANCE_PX = 8
MIN_FREE_RATIO = 0.30
STOP_RATIO = 0.20
MIN_CORRIDOR_RATIO = 0.25
SMOOTH_COLS = 9

# ---------- Controller (NORMALIZED -1.0 .. +1.0) ----------
# The ESP32 maps these to your calibrated maximums:
#   left 1.0 = 168 PWM, right 1.0 = 255 PWM
# so equal values on both sides = equal wheel effort.
BASE_SPEED = 0.55           # forward speed
TURN_SPEED = 0.50           # in-place rotation speed
MIN_LEFT = 0.45             # below this the left motor stalls (tune)
MIN_RIGHT = 0.45            # below this the right motor stalls (tune)
MAX_CMD = 0.75              # hard speed cap for safety
KP = 0.50
KD = 0.12

# ---------- Simple move / stop / turn controller ----------
DEADBAND = 0.25      # |error| below this = go straight
TURN_ON = 0.15       # seconds spinning per pulse
TURN_OFF = 0.20      # seconds stopped between pulses (camera catches up)
STOP_PAUSE = 0.30    # seconds stopped before turning when blocked

# ---------- Smarter behaviour ----------
DANGER_LABELS = {"person", "animal", "stairs", "stairway", "step", "escalator"}
DANGER_MARGIN = 14          # px of extra clearance around people/stairs
DANGER_CLOSE_RATIO = 0.04   # danger pixels in front-center that trigger WAIT
PERSON_CLEAR_TIME = 1.0     # seconds it must stay clear before moving again
DARK_LEVEL = 25             # mean brightness below this = stop

ENTER_ERR = 0.28            # start correcting heading above this error
EXIT_ERR = 0.12             # stop correcting below this error
SPIN_ERR = 0.60             # above this = spin pulses, below = smooth arc
ERR_SMOOTH = 0.6            # 0..1, higher = smoother but slower to react
TURN_ON_MIN = 0.10          # spin pulse length for small error (s)
TURN_ON_MAX = 0.25          # spin pulse length for large error (s)

SLOW_RATIO = 0.45           # clearance below this = creep mode
CREEP_ON = 0.20
CREEP_OFF = 0.20

RESUME_RATIO = 0.35         # clearance needed to stop scanning
SCAN_TIMEOUT = 12.0         # give up after this many seconds (LOST)
SCAN_REPEAT_WINDOW = 6.0    # blocked again within this time = wrong direction
RECOVER_TIME = 0.8          # spin time when stuck

STUCK_TIME = 2.0            # seconds of forward driving before checking
STUCK_DIFF = 2.0            # image change below this = stuck
