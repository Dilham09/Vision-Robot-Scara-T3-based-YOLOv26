"""
config.py
─────────────────────────────────────────────────────────────
Central configuration file for YOLO SCARA Pick & Place System.
Edit values here instead of modifying the main application.
"""

# ─── Model ────────────────────────────────────────────────
MODEL_PATH      = "models/best.pt"
CONF_THRESHOLD  = 0.4          # YOLO confidence threshold (0.1 – 1.0)

# ─── Camera ───────────────────────────────────────────────
CAMERA_SOURCE   = 0            # 0 = first USB camera; change to 1, 2, … if needed
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
CAMERA_FPS      = 30

# ─── Robot TCP/IP ─────────────────────────────────────────
ROBOT_IP        = "192.168.0.1"
ROBOT_PORT      = 20001

# ─── Grid Configuration ───────────────────────────────────
CLASS_GRID_CONFIG = {
    "pallet_pick":    {"cols": 3,  "rows": 8},
    "pallet_place_1": {"cols": 12, "rows": 2},
    "pallet_place_2": {"cols": 12, "rows": 2},
}
DEFAULT_GRID = {"cols": 3, "rows": 8}

# ─── Sequence Settings ────────────────────────────────────
DEFAULT_CMD_DELAY_MS = 500     # Delay between robot commands (ms)
DEFAULT_GRIPPER_OUTPUT = 0     # Robot digital output number for gripper

# ─── Circle Detection (HSV thresholds for white workpieces) ─
HSV_LOWER_WHITE = [0,   0,  180]
HSV_UPPER_WHITE = [180, 60, 255]
MIN_CIRCLE_AREA = 200          # Minimum contour area in pixels
MIN_CIRCULARITY = 0.75         # 0.0 (any shape) – 1.0 (perfect circle)
