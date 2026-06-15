<div align="center">

# 🤖 YOLO-Based Object Detection for SCARA T3 Robot Pick & Place Control

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-28A745?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Final%20Project-FFC107?style=for-the-badge)

**Final Project (Tugas Akhir) — Industrial Electrical Engineering**

*Perancangan dan Implementasi Object Detection Berbasis YOLO untuk Kontrol Robot SCARA T3 pada Proses Pick and Place*

**Dilham Hidayatul Fajri · 22130006 · Teknik Elektro Industri**

---

<!-- Replace the image below with an actual screenshot of the running application -->
![Application Screenshot](docs/images/app-screenshot.png)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Pallet Layout](#-pallet-layout)
- [Placement Modes](#-placement-modes)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🔍 Overview

This project implements a **real-time pick-and-place automation system** that integrates:

- **YOLOv8/v11 object detection** — detects workpiece positions on a source pallet via webcam
- **White circle detection** (HSV + contour analysis) — locates individual cylindrical workpieces within detected pallet regions
- **SCARA T3 Robot control** — sends `Jump Pallet()` commands over TCP/IP to move and grip workpieces
- **Tkinter GUI** — four-tab interface for live camera feed, communication logs, sequence monitoring, and manual placement planning

The system supports two placement modes:
- **Auto mode** — fills empty slots sequentially from left to right, top to bottom
- **Manual mode** — operator clicks target slots on an interactive grid to define a custom placement order

---

## ✨ Features

| Feature | Description |
|---|---|
| 📹 **Live Detection** | Capture frame → run YOLO → draw grid overlay on pallets |
| 🔵 **Circle Detection** | HSV masking + circularity filter to locate workpieces |
| 🤖 **Robot Control** | TCP/IP socket connection; send/receive commands in real time |
| 📡 **Communication Log** | Full log with filter, search, auto-scroll, and export to `.txt` |
| 🔄 **Sequence Engine** | Threaded pick-place loop with configurable delay and gripper output |
| 🗺️ **Placement Planner** | Interactive grid canvas — click slots to set custom placement order |
| 📊 **Statistics** | Detection count, uptime, last detected class |
| 🛡️ **Thread-safe UI** | `queue.Queue` message bus keeps Tkinter calls on the main thread |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tkinter GUI (Main Thread)             │
│  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Detection &  │ │ TCP/IP   │ │Pick&Place│ │Planner │ │
│  │   Control    │ │   Log    │ │ Monitor  │ │  Tab   │ │
│  └──────┬───────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└─────────┼──────────────┼────────────┼────────────┼──────┘
          │              │            │            │
    ┌─────▼──────┐  ┌────▼─────┐  ┌──▼───────────▼───┐
    │ YOLO Model │  │  Socket  │  │  Sequence Thread  │
    │(Ultralytics│  │  TCP/IP  │  │  (pick & place)   │
    │    .pt)    │  │ recv loop│  │                   │
    └─────┬──────┘  └────┬─────┘  └──────────┬────────┘
          │              │                    │
    ┌─────▼──────┐  ┌────▼─────────────────-─▼───┐
    │   OpenCV   │  │       SCARA T3 Robot         │
    │   Camera   │  │    192.168.0.1 : 20001       │
    └────────────┘  └─────────────────────────────┘
```

---

## 📁 Repository Structure

```
yolo-scara-pick-place/
│
├── main.py                    # Main application — run this
├── config.py                  # All configurable parameters
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── models/
│   └── best.pt                # Trained YOLO weights (tracked in repo)
│
├── docs/
│   ├── images/                # Screenshots for README
│   │   ├── app-screenshot.png
│   │   ├── tab-detection.png
│   │   ├── tab-tcp-log.png
│   │   ├── tab-monitor.png
│   │   └── tab-planner.png
│   └── manual/
│       └── Buku_Panduan_YOLO_Pick_Place.docx   # Full user manual
│
├── logs/                      # Exported TCP/IP logs (gitignored by default)
│
├── assets/                    # Icons, diagrams, or other static resources
│
└── .github/
    ├── workflows/
    │   └── python-check.yml   # GitHub Actions syntax check
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

---

## 📦 Requirements

### Hardware
| Component | Minimum Spec |
|---|---|
| Computer | Windows 10/11 64-bit, 8 GB RAM, Intel i5 / Ryzen 5 |
| Camera | USB Webcam 720p (1080p recommended) |
| Network | LAN / WiFi — same subnet as robot |
| Robot | SCARA T3 with TCP/IP remote control enabled |
| GPU *(optional)* | NVIDIA CUDA GPU for faster inference |

### Software
- Python **3.10** or **3.11** (3.12 not tested)
- pip packages — see [`requirements.txt`](requirements.txt)

---

## 🚀 Installation

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/yolo-scara-pick-place.git
cd yolo-scara-pick-place
```

### 2 — Create a virtual environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA separately before running:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4 — Place your model

Copy your trained YOLO weights into the `models/` folder:

```
models/
└── best.pt   ← your file goes here
```

If your model is stored elsewhere, edit `MODEL_PATH` in [`config.py`](config.py).

### 5 — Run

```bash
python main.py
```

---

## ⚙️ Configuration

All parameters are centralized in [`config.py`](config.py). Edit this file — **do not** hardcode values inside `main.py`.

```python
# config.py (key settings)

MODEL_PATH      = "models/best.pt"     # Path to YOLO weights
CONF_THRESHOLD  = 0.4                  # Detection confidence (0.1–1.0)

CAMERA_SOURCE   = 0                    # USB camera index
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720

ROBOT_IP        = "192.168.0.1"        # ← change to your robot's IP
ROBOT_PORT      = 20001

DEFAULT_CMD_DELAY_MS   = 500           # ms between robot commands
DEFAULT_GRIPPER_OUTPUT = 0             # robot digital output for gripper
```

> The Robot IP can also be changed at runtime from the GUI — no restart needed.

### Grid Configuration

```python
CLASS_GRID_CONFIG = {
    "pallet_pick":    {"cols": 3,  "rows": 8},   # 24 slots
    "pallet_place_1": {"cols": 12, "rows": 2},   # 24 slots
    "pallet_place_2": {"cols": 12, "rows": 2},   # 24 slots
}
```

Your YOLO model must output class names containing `"pick"` or `"place_1"` for the grid overlay to activate.

---

## 🖥️ Usage

### Standard workflow (Auto mode)

```
1. Run main.py
2. Wait for "Model: Ready ✅"
3. Enter Robot IP → click "Connect Robot"
4. Position pallets under the camera
5. Click "📸 Capture & Detect Pallets"
6. Verify grid overlay is accurate
7. Click "▶ Start Sequence"
8. Monitor progress in the Pick & Place Monitor tab
9. Robot returns Home automatically when done
```

### Manual placement workflow

```
1. Complete steps 1–6 above
2. Open tab  🗺️ Placement Planner
3. Select  🖱️ Manual mode
4. Click slots in the desired placement order
5. Click "✅ Terapkan ke Queue"
6. Return to Detection & Control → click "▶ Start Sequence"
```

### Robot commands sent per workpiece

```
Jump Pallet(1, col, row)   → move to pick position
On  <output>               → activate gripper (pick)
Jump Pallet(2, col, row)   → move to place position
Off <output>               → release gripper (place)
```

After all tasks complete:

```
Home                       → return to home position
```

---

## 📐 Pallet Layout

### Pallet 1 — Pick (Source)
Grid: **3 columns × 8 rows** — slot numbering **(col, row)** from top-left

```
     Col 1    Col 2    Col 3
Row 1 [ 1,1 ] [ 2,1 ] [ 3,1 ]
Row 2 [ 1,2 ] [ 2,2 ] [ 3,2 ]
Row 3 [ 1,3 ] [ 2,3 ] [ 3,3 ]
Row 4 [ 1,4 ] [ 2,4 ] [ 3,4 ]
Row 5 [ 1,5 ] [ 2,5 ] [ 3,5 ]
Row 6 [ 1,6 ] [ 2,6 ] [ 3,6 ]
Row 7 [ 1,7 ] [ 2,7 ] [ 3,7 ]
Row 8 [ 1,8 ] [ 2,8 ] [ 3,8 ]
                               Total: 24 slots
```

### Pallet 2 — Place (Destination)
Grid: **12 columns × 2 rows** — same slot numbering convention

```
  C1    C2    C3    C4    C5    C6    C7    C8    C9   C10   C11   C12
R1 [1,1][2,1][3,1][4,1][5,1][6,1][7,1][8,1][9,1][10,1][11,1][12,1]
R2 [1,2][2,2][3,2][4,2][5,2][6,2][7,2][8,2][9,2][10,2][11,2][12,2]
                                                             Total: 24 slots
```

> **Auto mode** fills Row 1 left → right, then Row 2 left → right.  
> **Manual mode** lets you define any order by clicking slots in the Placement Planner tab.

---

## 🗺️ Placement Modes

### Auto Mode *(default)*

The system scans `pallet_place` for already-occupied slots (detected visually), then assigns the next available empty slot to each picked workpiece — in left-to-right, top-to-bottom order.

### Manual Mode

1. Open **Placement Planner** tab after detection
2. Click each slot in the order you want workpieces placed — numbers appear in cells
3. Use **"Isi Otomatis"** to pre-fill auto order, then rearrange by clicking
4. Click **"✅ Terapkan ke Queue"** to commit
5. Start the sequence

| Slot Color | Meaning |
|---|---|
| 🟩 Green | Already occupied (detected) |
| 🔵 Blue + number | Selected manual slot (number = order) |
| ⬜ Gray | Empty, available |

---

## 🔧 Troubleshooting

### ⚠️ Critical: Robot places two workpieces in the same slot

This happens when the system doesn't detect a workpiece already placed in a slot, so it assigns that slot again.

**Root causes & fixes:**

| Cause | Fix |
|---|---|
| HSV detection misses placed workpiece | Lower `MIN_CIRCLE_AREA` to `100`, lower `MIN_CIRCULARITY` to `0.65` in `config.py` |
| Poor lighting on place pallet | Add even LED lighting above the place pallet area |
| Camera vibration during robot motion | Use a rigid camera mount; isolate from robot vibration |
| Confidence threshold too high | Lower `CONF_THRESHOLD` to `0.3` |
| No re-detection between tasks | **Use Manual mode** — slot assignments are fixed, not vision-dependent |

**Immediate workaround:** Switch to **Manual mode** in the Placement Planner tab. Manual mode uses a predefined slot list that never re-evaluates vision, eliminating double-fill from detection errors.

---

### Other Common Issues

<details>
<summary><strong>Robot not connecting</strong></summary>

- Verify robot is powered on and in Remote/Auto mode
- Confirm the IP in `config.py` or GUI matches the robot
- Check both devices are on the same subnet (`ping 192.168.0.1`)
- Temporarily disable Windows Firewall for testing

</details>

<details>
<summary><strong>Camera not detected</strong></summary>

- Try `CAMERA_SOURCE = 1` or `2` in `config.py`
- Check Windows Device Manager for camera index
- Ensure no other app (Teams, Zoom) is using the camera

</details>

<details>
<summary><strong>Model fails to load</strong></summary>

- Check `MODEL_PATH` in `config.py` points to the correct location
- Verify `best.pt` exists and is not corrupted (re-copy if needed)
- Ensure `ultralytics` is installed: `pip install ultralytics`

</details>

<details>
<summary><strong>Grid overlay misaligned</strong></summary>

- Mount the camera directly overhead, perpendicular (90°) to pallet surface
- Avoid wide-angle lenses — barrel distortion shifts grid cells
- Re-run Capture & Detect after any camera movement

</details>

<details>
<summary><strong>Robot moves too fast / slips</strong></summary>

- Increase `DEFAULT_CMD_DELAY_MS` to `800`–`1500` in `config.py`  
  or adjust in the GUI under **Pick & Place Monitor → Sequence Settings**

</details>

<details>
<summary><strong>Queue empty after detection (Manual mode)</strong></summary>

After detection, if mode is Manual, the queue is **not** built automatically.  
Go to **Placement Planner** → select slots → click **"✅ Terapkan ke Queue"** before starting.

</details>

---

## 📚 Documentation

| Document | Location |
|---|---|
| Full User Manual (Bahasa Indonesia) | [`docs/manual/Buku_Panduan_YOLO_Pick_Place.docx`](docs/manual/Buku_Panduan_YOLO_Pick_Place.docx) |
| App screenshots | [`docs/images/`](docs/images/) |
| Configuration reference | [`config.py`](config.py) |

---

## 🛣️ Roadmap

- [ ] Re-detection verification before each Place step (prevent double-fill in Auto mode)
- [ ] Live video stream (continuous camera feed, not single-frame capture)
- [ ] Export queue plan as CSV before execution
- [ ] Support for multiple pallet place destinations in a single session
- [ ] Configurable HSV thresholds from GUI (no code edit needed)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 / YOLO11 framework
- [OpenCV](https://opencv.org/) — computer vision library
- Python `tkinter` — GUI framework (stdlib)
- SCARA T3 Robot — Epson / Denso compatible TCP/IP protocol

---

<div align="center">

Made with ❤️ as part of **Tugas Akhir — Teknik Elektro Industri**

*Dilham Hidayatul Fajri · 22130006*

</div>
