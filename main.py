import cv2
import numpy as np
from pathlib import Path
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import socket
import threading
import time
import queue
from datetime import datetime

# Ultralytics YOLO (v8/v11/v26+)
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
#  Grid config per class
# ─────────────────────────────────────────────
CLASS_GRID_CONFIG = {
    "pallet_pick":    {"cols": 3,  "rows": 8},
    "pallet_place_1": {"cols": 12, "rows": 2},
    "pallet_place_2": {"cols": 12, "rows": 2},
}
DEFAULT_GRID = {"cols": 3, "rows": 8}


class ModernYOLOGUI:
    def __init__(self):
        # ===================== Config =====================
        self.MODEL_PATH = "models/best.pt"
        self.CONF_THRESHOLD = 0.4
        self.WIDTH, self.HEIGHT = 1280, 720
        self.SOURCE = 0
        self.ROBOT_IP = "192.168.0.1"
        self.ROBOT_PORT = 20001

        # Variables
        self.model = None
        self.cap = None
        self.sock = None
        self.is_connected = False
        self.detection_count = 0
        self.start_time = time.time()

        # Thread-safe communication
        self.message_queue = queue.Queue()

        # TCP/IP log storage
        self.tcp_log = []
        self.tcp_log_lock = threading.Lock()

        # ── Pick & Place State ─────────────────────────────
        # Menyimpan slot yang terisi di pallet 1 (set of (col, row) tuples)
        self.pallet1_occupied = set()
        # Menyimpan slot yang terisi di pallet 2 (set of (col, row) tuples)
        self.pallet2_occupied = set()
        # Queue antrian tugas pick & place: list of (pick_coord, place_coord)
        self.pick_place_queue = []
        # Flag apakah sedang menjalankan sekuens
        self.is_running_sequence = False
        # Lock untuk pick_place_queue
        self.pp_lock = threading.Lock()

        self.setup_gui()
        self.start_message_processor()
        self.load_model()
        self.setup_camera()

    # ──────────────────────────────────────────
    #  GUI Setup
    # ──────────────────────────────────────────
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title(
            "Perancangan dan Implementasi Object Detection Berbasis YOLO "
            "untuk Kontrol Robot SCARA T3 pada Proses Pick and Place"
        )
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')

        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel',    background='#1a1a1a', foreground='#00d4ff',
                        font=('Segoe UI', 16, 'bold'))
        style.configure('Subtitle.TLabel', background='#1a1a1a', foreground='#ffffff',
                        font=('Segoe UI', 11))
        style.configure('Modern.TButton',  font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.configure('Success.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('Danger.TButton',  font=('Segoe UI', 10, 'bold'))
        style.configure('Warning.TButton', font=('Segoe UI', 10, 'bold'))

        style.map('Modern.TButton',
                  background=[('active', '#0056b3'), ('!active', '#007acc')])
        style.map('Success.TButton',
                  background=[('active', '#218838'), ('!active', '#28a745')])
        style.map('Danger.TButton',
                  background=[('active', '#c82333'), ('!active', '#dc3545')])
        style.map('Warning.TButton',
                  background=[('active', '#d39e00'), ('!active', '#ffc107')])

        # ── Tab styling ──────────────────────────
        style.configure('TNotebook',           background='#1a1a1a', borderwidth=0)
        style.configure('TNotebook.Tab',
                        background='#2d2d2d', foreground='#aaaaaa',
                        font=('Segoe UI', 11, 'bold'),
                        padding=[20, 8])
        style.map('TNotebook.Tab',
                  background=[('selected', '#1a1a1a'), ('active', '#3a3a3a')],
                  foreground=[('selected', '#00d4ff'), ('active', '#ffffff')])

        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_frame, bg='#1a1a1a')
        header_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(
            header_frame,
            text="Tugas Akhir\n"
                 "Perancangan dan Implementasi Object Detection Berbasis YOLO\n"
                 "untuk Kontrol Robot SCARA T3 pada Proses Pick and Place",
            style='Title.TLabel', justify='center'
        ).pack()
        ttk.Label(
            header_frame,
            text="Dilham Hidayatul Fajri  |  22130006  |  Teknik Elektro Industri",
            style='Subtitle.TLabel', justify='center'
        ).pack(pady=(4, 0))

        # ── Notebook (Tab Container) ─────────────
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1 — Main GUI
        self.tab_main = tk.Frame(self.notebook, bg='#1a1a1a')
        self.notebook.add(self.tab_main, text="📹  Detection & Control")

        # Tab 2 — TCP/IP Log
        self.tab_tcp = tk.Frame(self.notebook, bg='#1a1a1a')
        self.notebook.add(self.tab_tcp, text="📡  TCP/IP Communication Log")

        # Tab 3 — Pick & Place Monitor
        self.tab_pp = tk.Frame(self.notebook, bg='#1a1a1a')
        self.notebook.add(self.tab_pp, text="🔄  Pick & Place Monitor")

        self.build_tab_main()
        self.build_tab_tcp()
        self.build_tab_pp()

        self.create_status_bar(main_frame)

    # ──────────────────────────────────────────
    #  Tab 1 — Detection & Control
    # ──────────────────────────────────────────
    def build_tab_main(self):
        content_frame = tk.Frame(self.tab_main, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)

        self.create_system_status(left_panel)
        self.create_robot_controls(left_panel)
        self.create_detection_settings(left_panel)
        self.create_pick_place_controls(left_panel)
        self.create_grid_info(left_panel)
        self.create_statistics(left_panel)

        # Right panel — camera
        right_panel = tk.Frame(content_frame, bg='#2d2d2d')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        camera_header = tk.Frame(right_panel, bg='#2d2d2d')
        camera_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        ttk.Label(camera_header, text="📹 Camera Feed", style='Subtitle.TLabel').pack(side=tk.LEFT)

        camera_frame = tk.Frame(right_panel, bg='#000000', relief=tk.SUNKEN, bd=2)
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.camera_panel = tk.Label(
            camera_frame, bg='#000000',
            text="🎥\n\nCamera Ready\n\nClick 'Capture & Detect' to scan pallets",
            fg='#666666', font=('Segoe UI', 14)
        )
        self.camera_panel.pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────
    #  Tab 2 — TCP/IP Communication Log
    # ──────────────────────────────────────────
    def build_tab_tcp(self):
        ctrl_bar = tk.Frame(self.tab_tcp, bg='#1a1a1a')
        ctrl_bar.pack(fill=tk.X, padx=20, pady=(15, 5))

        ttk.Label(ctrl_bar, text="📡 TCP/IP Communication Log",
                  style='Title.TLabel').pack(side=tk.LEFT)

        btn_frame = tk.Frame(ctrl_bar, bg='#1a1a1a')
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="🗑️  Clear Log",
                   command=self.clear_tcp_log,
                   style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾  Export Log",
                   command=self.export_tcp_log,
                   style='Modern.TButton').pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_frame, text="Auto-scroll",
                       variable=self.auto_scroll_var,
                       bg='#1a1a1a', fg='#ffffff', selectcolor='#2d2d2d',
                       activebackground='#1a1a1a', activeforeground='#ffffff',
                       font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=10)

        filter_frame = tk.Frame(self.tab_tcp, bg='#2d2d2d')
        filter_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(filter_frame, text="  Filter:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, pady=6)

        self.filter_var = tk.StringVar(value="ALL")
        for opt, lbl in [("ALL", "All"), ("SENT", "📤 Sent"), ("RECV", "📥 Received"), ("INFO", "ℹ️ Info"), ("ERR", "❌ Error")]:
            tk.Radiobutton(filter_frame, text=lbl, variable=self.filter_var,
                           value=opt, bg='#2d2d2d', fg='#ffffff',
                           selectcolor='#404040', activebackground='#2d2d2d',
                           activeforeground='#00d4ff', font=('Segoe UI', 9),
                           command=self.refresh_tcp_display).pack(side=tk.LEFT, padx=8, pady=6)

        tk.Label(filter_frame, text="  Search:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self.refresh_tcp_display())
        tk.Entry(filter_frame, textvariable=self.search_var, width=20,
                 bg='#404040', fg='#ffffff', insertbackground='#ffffff',
                 font=('Segoe UI', 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=8, pady=6)

        stats_frame = tk.Frame(self.tab_tcp, bg='#252525')
        stats_frame.pack(fill=tk.X, padx=20, pady=(0, 5))

        self.stat_sent  = tk.Label(stats_frame, text="📤 Sent: 0",    bg='#252525', fg='#28a745', font=('Segoe UI', 9, 'bold'))
        self.stat_recv  = tk.Label(stats_frame, text="📥 Received: 0",bg='#252525', fg='#00d4ff', font=('Segoe UI', 9, 'bold'))
        self.stat_err   = tk.Label(stats_frame, text="❌ Errors: 0",  bg='#252525', fg='#dc3545', font=('Segoe UI', 9, 'bold'))
        self.stat_total = tk.Label(stats_frame, text="Total: 0",      bg='#252525', fg='#aaaaaa', font=('Segoe UI', 9))

        for w in [self.stat_sent, self.stat_recv, self.stat_err, self.stat_total]:
            w.pack(side=tk.LEFT, padx=15, pady=5)

        log_frame = tk.Frame(self.tab_tcp, bg='#0d0d0d', relief=tk.SUNKEN, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        v_scroll = tk.Scrollbar(log_frame, orient=tk.VERTICAL, bg='#2d2d2d', troughcolor='#1a1a1a')
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = tk.Scrollbar(log_frame, orient=tk.HORIZONTAL, bg='#2d2d2d', troughcolor='#1a1a1a')
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.tcp_log_text = tk.Text(
            log_frame, bg='#0d0d0d', fg='#d4d4d4',
            font=('Consolas', 10), state=tk.DISABLED, wrap=tk.NONE,
            yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set,
            selectbackground='#264f78', insertbackground='#ffffff'
        )
        self.tcp_log_text.pack(fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.tcp_log_text.yview)
        h_scroll.config(command=self.tcp_log_text.xview)

        self.tcp_log_text.tag_configure("SENT",      foreground='#4ec9b0')
        self.tcp_log_text.tag_configure("RECV",      foreground='#9cdcfe')
        self.tcp_log_text.tag_configure("INFO",      foreground='#dcdcaa')
        self.tcp_log_text.tag_configure("ERR",       foreground='#f48771')
        self.tcp_log_text.tag_configure("TIMESTAMP", foreground='#569cd6')
        self.tcp_log_text.tag_configure("BRACKET",   foreground='#808080')
        self.tcp_log_text.tag_configure("HIGHLIGHT", background='#3a3a00', foreground='#ffff00')

        send_frame = tk.LabelFrame(self.tab_tcp, text="✏️  Manual Send Command",
                                   bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 9, 'bold'))
        send_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        inner = tk.Frame(send_frame, bg='#2d2d2d')
        inner.pack(fill=tk.X, padx=10, pady=8)

        self.manual_cmd_var = tk.StringVar()
        cmd_entry = tk.Entry(inner, textvariable=self.manual_cmd_var,
                             bg='#404040', fg='#ffffff', insertbackground='#ffffff',
                             font=('Consolas', 10), relief=tk.FLAT)
        cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        cmd_entry.bind("<Return>", lambda e: self.manual_send())

        ttk.Button(inner, text="📤 Send", command=self.manual_send,
                   style='Success.TButton').pack(side=tk.LEFT, padx=(8, 0))

    # ──────────────────────────────────────────
    #  Tab 3 — Pick & Place Monitor
    # ──────────────────────────────────────────
    def build_tab_pp(self):
        ctrl_bar = tk.Frame(self.tab_pp, bg='#1a1a1a')
        ctrl_bar.pack(fill=tk.X, padx=20, pady=(15, 5))

        ttk.Label(ctrl_bar, text="🔄 Pick & Place Monitor",
                  style='Title.TLabel').pack(side=tk.LEFT)

        # ── Queue info ──
        info_frame = tk.Frame(self.tab_pp, bg='#252525')
        info_frame.pack(fill=tk.X, padx=20, pady=5)

        self.pp_queue_label = tk.Label(info_frame, text="📋 Queue: 0 tasks",
                                       bg='#252525', fg='#ffc107', font=('Segoe UI', 10, 'bold'))
        self.pp_queue_label.pack(side=tk.LEFT, padx=15, pady=8)

        self.pp_status_label = tk.Label(info_frame, text="⏸️ Idle",
                                        bg='#252525', fg='#aaaaaa', font=('Segoe UI', 10, 'bold'))
        self.pp_status_label.pack(side=tk.LEFT, padx=15)

        self.pp_progress_label = tk.Label(info_frame, text="0 / 0 selesai",
                                          bg='#252525', fg='#28a745', font=('Segoe UI', 10, 'bold'))
        self.pp_progress_label.pack(side=tk.RIGHT, padx=15)

        # ── Task list ──
        list_frame = tk.Frame(self.tab_pp, bg='#0d0d0d', relief=tk.SUNKEN, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        v_scroll2 = tk.Scrollbar(list_frame, orient=tk.VERTICAL, bg='#2d2d2d', troughcolor='#1a1a1a')
        v_scroll2.pack(side=tk.RIGHT, fill=tk.Y)

        self.pp_log_text = tk.Text(
            list_frame, bg='#0d0d0d', fg='#d4d4d4',
            font=('Consolas', 10), state=tk.DISABLED, wrap=tk.NONE,
            yscrollcommand=v_scroll2.set,
            selectbackground='#264f78'
        )
        self.pp_log_text.pack(fill=tk.BOTH, expand=True)
        v_scroll2.config(command=self.pp_log_text.yview)

        self.pp_log_text.tag_configure("HEADER",  foreground='#00d4ff', font=('Consolas', 10, 'bold'))
        self.pp_log_text.tag_configure("PICK",    foreground='#ffc107')
        self.pp_log_text.tag_configure("PLACE",   foreground='#4ec9b0')
        self.pp_log_text.tag_configure("DONE",    foreground='#28a745')
        self.pp_log_text.tag_configure("PENDING", foreground='#808080')
        self.pp_log_text.tag_configure("CURRENT", foreground='#ffffff', background='#1a4a1a')

        # ── Delay setting ──
        delay_frame = tk.LabelFrame(self.tab_pp, text="⚙️  Sequence Settings",
                                    bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 9, 'bold'))
        delay_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        inner_d = tk.Frame(delay_frame, bg='#2d2d2d')
        inner_d.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(inner_d, text="Delay antar perintah (ms):", bg='#2d2d2d', fg='#ffffff',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.cmd_delay_var = tk.IntVar(value=500)
        tk.Spinbox(inner_d, from_=100, to=5000, increment=100, textvariable=self.cmd_delay_var,
                   width=6, bg='#404040', fg='#ffffff', insertbackground='#ffffff',
                   font=('Segoe UI', 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=10)

        tk.Label(inner_d, text="ms", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)

        # ── ON output setting ──
        tk.Label(inner_d, text="    Output ON robot:", bg='#2d2d2d', fg='#ffffff',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(20, 0))
        self.on_output_var = tk.IntVar(value=0)
        tk.Spinbox(inner_d, from_=0, to=15, increment=1, textvariable=self.on_output_var,
                   width=4, bg='#404040', fg='#ffffff', insertbackground='#ffffff',
                   font=('Segoe UI', 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

    # ──────────────────────────────────────────
    #  Left-panel widgets (Tab 1)
    # ──────────────────────────────────────────
    def create_system_status(self, parent):
        frame = tk.LabelFrame(parent, text="🔧 System Status",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)

        for label_text, attr in [("Model:", "model_status"),
                                  ("Camera:", "camera_status"),
                                  ("Robot:", "robot_status")]:
            row = tk.Frame(frame, bg='#2d2d2d')
            row.pack(fill=tk.X, padx=10, pady=4)
            tk.Label(row, text=label_text, bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT)
            lbl = tk.Label(row, text="Loading..." if attr == "model_status" else "Disconnected",
                           bg='#2d2d2d',
                           fg='#ffc107' if attr == "model_status" else '#dc3545')
            lbl.pack(side=tk.RIGHT)
            setattr(self, attr, lbl)

    def create_robot_controls(self, parent):
        frame = tk.LabelFrame(parent, text="🤖 Robot Control",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)

        conn_frame = tk.Frame(frame, bg='#2d2d2d')
        conn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.connect_btn = ttk.Button(conn_frame, text="🔌 Connect Robot",
                                      command=self.connect_robot, style='Success.TButton')
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.disconnect_btn = ttk.Button(conn_frame, text="🔌 Disconnect",
                                         command=self.disconnect_robot,
                                         style='Danger.TButton', state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.RIGHT)

        ip_frame = tk.Frame(frame, bg='#2d2d2d')
        ip_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(ip_frame, text="Robot IP:", bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT)
        self.ip_entry = tk.Entry(ip_frame, width=15, bg='#404040', fg='#ffffff',
                                 insertbackground='#ffffff')
        self.ip_entry.pack(side=tk.RIGHT)
        self.ip_entry.insert(0, self.ROBOT_IP)

    def create_detection_settings(self, parent):
        frame = tk.LabelFrame(parent, text="🎯 Detection Settings",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)

        conf_frame = tk.Frame(frame, bg='#2d2d2d')
        conf_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(conf_frame, text="Confidence:", bg='#2d2d2d', fg='#ffffff').pack(anchor=tk.W)

        self.conf_var = tk.DoubleVar(value=self.CONF_THRESHOLD)
        tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.conf_var, bg='#2d2d2d', fg='#ffffff', highlightthickness=0,
                 command=self.update_confidence).pack(fill=tk.X, pady=5)

        ttk.Button(frame, text="📸 Capture & Detect Pallets",
                   command=self.capture_and_process,
                   style='Success.TButton').pack(fill=tk.X, padx=10, pady=10)

    def create_pick_place_controls(self, parent):
        """Panel kontrol khusus Pick & Place."""
        frame = tk.LabelFrame(parent, text="🔄 Pick & Place Control",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)

        # Start / Stop buttons
        btn_row = tk.Frame(frame, bg='#2d2d2d')
        btn_row.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.start_pp_btn = ttk.Button(btn_row, text="▶ Start Sequence",
                                       command=self.start_pick_place_sequence,
                                       style='Success.TButton')
        self.start_pp_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_pp_btn = ttk.Button(btn_row, text="⏹ Stop",
                                      command=self.stop_pick_place_sequence,
                                      style='Danger.TButton', state=tk.DISABLED)
        self.stop_pp_btn.pack(side=tk.LEFT)

        # Reset queue button
        ttk.Button(frame, text="🗑️  Reset Queue & Pallet State",
                   command=self.reset_pick_place,
                   style='Warning.TButton').pack(fill=tk.X, padx=10, pady=(0, 10))

        # Info label
        self.pp_info_label = tk.Label(frame,
                                      text="Tekan 'Capture & Detect' dulu,\nlalu 'Start Sequence'",
                                      bg='#2d2d2d', fg='#aaaaaa', font=('Segoe UI', 8),
                                      justify='center')
        self.pp_info_label.pack(pady=(0, 8))

    def create_grid_info(self, parent):
        frame = tk.LabelFrame(parent, text="📐 Grid Configuration",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=5)

        info = [
            ("pallet_pick",    "3 × 8  (Col × Row)"),
            ("pallet_place_1", "12 × 2 (Col × Row)"),
            ("pallet_place_2", "12 × 2 (Col × Row)"),
        ]
        for cls, desc in info:
            row = tk.Frame(frame, bg='#2d2d2d')
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=cls, bg='#2d2d2d', fg='#00d4ff',
                     font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg='#2d2d2d', fg='#ffffff',
                     font=('Segoe UI', 9)).pack(side=tk.RIGHT)

    def create_statistics(self, parent):
        frame = tk.LabelFrame(parent, text="📊 Statistics",
                              bg='#2d2d2d', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)

        det_frame = tk.Frame(frame, bg='#2d2d2d')
        det_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(det_frame, text="Detections:", bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT)
        self.detection_label = tk.Label(det_frame, text="0", bg='#2d2d2d', fg='#00d4ff',
                                        font=('Segoe UI', 10, 'bold'))
        self.detection_label.pack(side=tk.RIGHT)

        cls_frame = tk.Frame(frame, bg='#2d2d2d')
        cls_frame.pack(fill=tk.X, padx=10, pady=3)
        tk.Label(cls_frame, text="Last Class:", bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT)
        self.last_class_label = tk.Label(cls_frame, text="-", bg='#2d2d2d', fg='#ffc107',
                                         font=('Segoe UI', 9, 'bold'))
        self.last_class_label.pack(side=tk.RIGHT)

        time_frame = tk.Frame(frame, bg='#2d2d2d')
        time_frame.pack(fill=tk.X, padx=10, pady=(3, 10))
        tk.Label(time_frame, text="Uptime:", bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT)
        self.uptime_label = tk.Label(time_frame, text="00:00:00", bg='#2d2d2d', fg='#28a745',
                                     font=('Segoe UI', 10, 'bold'))
        self.uptime_label.pack(side=tk.RIGHT)

        ttk.Button(frame, text="🔄 Reset Stats", command=self.reset_stats,
                   style='Modern.TButton').pack(pady=10)

    def create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg='#404040', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_text = tk.Label(status_frame, text="Ready",
                                    bg='#404040', fg='#ffffff', font=('Segoe UI', 9))
        self.status_text.pack(side=tk.LEFT, padx=10, pady=5)

        self.time_label = tk.Label(status_frame, text="", bg='#404040', fg='#ffffff',
                                   font=('Segoe UI', 9))
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        self.update_time()

    # ──────────────────────────────────────────
    #  TCP Log helpers
    # ──────────────────────────────────────────
    def _add_tcp_log(self, msg_type: str, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = {"ts": ts, "type": msg_type, "msg": message}
        with self.tcp_log_lock:
            self.tcp_log.append(entry)
        self.root.after(0, self._update_tcp_stats_and_display, entry)

    def _update_tcp_stats_and_display(self, entry):
        with self.tcp_log_lock:
            sent  = sum(1 for e in self.tcp_log if e["type"] == "SENT")
            recv  = sum(1 for e in self.tcp_log if e["type"] == "RECV")
            err   = sum(1 for e in self.tcp_log if e["type"] == "ERR")
            total = len(self.tcp_log)

        self.stat_sent.config( text=f"📤 Sent: {sent}")
        self.stat_recv.config( text=f"📥 Received: {recv}")
        self.stat_err.config(  text=f"❌ Errors: {err}")
        self.stat_total.config(text=f"Total: {total}")

        filt   = self.filter_var.get()
        search = self.search_var.get().lower()
        if filt != "ALL" and entry["type"] != filt:
            return
        if search and search not in entry["msg"].lower():
            return
        self._append_log_line(entry)

    def _append_log_line(self, entry):
        icons = {"SENT": "📤", "RECV": "📥", "INFO": "ℹ️ ", "ERR": "❌"}
        icon  = icons.get(entry["type"], "  ")
        self.tcp_log_text.config(state=tk.NORMAL)
        self.tcp_log_text.insert(tk.END, f"[{entry['ts']}]", "TIMESTAMP")
        self.tcp_log_text.insert(tk.END, " [", "BRACKET")
        self.tcp_log_text.insert(tk.END, f"{entry['type']:4s}", entry["type"])
        self.tcp_log_text.insert(tk.END, "] ", "BRACKET")
        self.tcp_log_text.insert(tk.END, f"{icon} {entry['msg']}\n", entry["type"])
        self.tcp_log_text.config(state=tk.DISABLED)
        if self.auto_scroll_var.get():
            self.tcp_log_text.see(tk.END)

    def refresh_tcp_display(self):
        filt   = self.filter_var.get()
        search = self.search_var.get().lower()
        self.tcp_log_text.config(state=tk.NORMAL)
        self.tcp_log_text.delete("1.0", tk.END)
        with self.tcp_log_lock:
            entries = list(self.tcp_log)
        for entry in entries:
            if filt != "ALL" and entry["type"] != filt:
                continue
            if search and search not in entry["msg"].lower():
                continue
            self._append_log_line(entry)
        self.tcp_log_text.config(state=tk.DISABLED)

    def clear_tcp_log(self):
        with self.tcp_log_lock:
            self.tcp_log.clear()
        self.tcp_log_text.config(state=tk.NORMAL)
        self.tcp_log_text.delete("1.0", tk.END)
        self.tcp_log_text.config(state=tk.DISABLED)
        self.stat_sent.config( text="📤 Sent: 0")
        self.stat_recv.config( text="📥 Received: 0")
        self.stat_err.config(  text="❌ Errors: 0")
        self.stat_total.config(text="Total: 0")
        self._add_tcp_log("INFO", "Log cleared by user")

    def export_tcp_log(self):
        import tkinter.filedialog as fd
        path = fd.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export TCP/IP Log"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                with self.tcp_log_lock:
                    for e in self.tcp_log:
                        f.write(f"[{e['ts']}] [{e['type']:4s}] {e['msg']}\n")
            self._add_tcp_log("INFO", f"Log exported to: {path}")
        except Exception as ex:
            self._add_tcp_log("ERR", f"Export failed: {ex}")

    def manual_send(self):
        cmd = self.manual_cmd_var.get().strip()
        if not cmd:
            return
        self.manual_cmd_var.set("")
        self.send_robot_command(cmd)

    # ──────────────────────────────────────────
    #  Thread-safe messaging
    # ──────────────────────────────────────────
    def start_message_processor(self):
        def process_messages():
            try:
                while True:
                    message_type, data = self.message_queue.get_nowait()
                    if message_type == "model_status":
                        status, color, text = data
                        self.model_status.config(text=status, fg=color)
                        self.status_text.config(text=text)
                    elif message_type == "camera_status":
                        status, color, text = data
                        self.camera_status.config(text=status, fg=color)
                        self.status_text.config(text=text)
                    elif message_type == "robot_status":
                        status, color, text, cs, ds = data
                        self.robot_status.config(text=status, fg=color)
                        self.connect_btn.config(state=cs)
                        self.disconnect_btn.config(state=ds)
                        self.status_text.config(text=text)
                    elif message_type == "status_text":
                        self.status_text.config(text=data)
                    elif message_type == "error":
                        messagebox.showerror("Error", data)
                    elif message_type == "update_pp_monitor":
                        self._refresh_pp_monitor()
            except queue.Empty:
                pass
            self.root.after(100, process_messages)

        self.root.after(100, process_messages)

    def queue_message(self, message_type, data):
        self.message_queue.put((message_type, data))

    # ──────────────────────────────────────────
    #  Model & Camera
    # ──────────────────────────────────────────
    def load_model(self):
        def load():
            try:
                self.queue_message("status_text", "Loading YOLO model (ultralytics)...")
                self._add_tcp_log("INFO", "Loading YOLO model...")
                self.model = YOLO(self.MODEL_PATH)
                self.model.conf = self.CONF_THRESHOLD
                self.queue_message("model_status", ("Ready ✅", '#28a745', "Model loaded successfully"))
                self._add_tcp_log("INFO", "YOLO model loaded successfully")
            except Exception as e:
                self.queue_message("model_status", ("Error ❌", '#dc3545', f"Model load failed: {e}"))
                self.queue_message("error", f"Failed to load model:\n{e}")
                self._add_tcp_log("ERR", f"Model load failed: {e}")

        threading.Thread(target=load, daemon=True).start()

    def setup_camera(self):
        def setup():
            try:
                self._add_tcp_log("INFO", f"Initializing camera (source={self.SOURCE})...")
                self.cap = cv2.VideoCapture(self.SOURCE, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.queue_message("camera_status", ("Connected ✅", '#28a745', "Camera initialized"))
                    self._add_tcp_log("INFO", "Camera initialized successfully")
                else:
                    self.queue_message("camera_status", ("Error ❌", '#dc3545', "Camera initialization failed"))
                    self._add_tcp_log("ERR", "Camera initialization failed")
            except Exception as e:
                self.queue_message("camera_status", ("Error ❌", '#dc3545', f"Camera error: {e}"))
                self._add_tcp_log("ERR", f"Camera error: {e}")

        threading.Thread(target=setup, daemon=True).start()

    # ──────────────────────────────────────────
    #  Robot / TCP
    # ──────────────────────────────────────────
    def connect_robot(self):
        def connect():
            try:
                self.ROBOT_IP = self.ip_entry.get()
                self._add_tcp_log("INFO", f"Connecting to {self.ROBOT_IP}:{self.ROBOT_PORT}...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(None)
                self.sock.connect((self.ROBOT_IP, self.ROBOT_PORT))
                self.is_connected = True
                self.queue_message("robot_status",
                    ("Connected ✅", '#28a745', f"Connected to robot at {self.ROBOT_IP}",
                     tk.DISABLED, tk.NORMAL))
                self._add_tcp_log("INFO", f"TCP connection established → {self.ROBOT_IP}:{self.ROBOT_PORT}")
                threading.Thread(target=self._recv_loop, daemon=True).start()
            except Exception as e:
                self.is_connected = False
                if self.sock:
                    self.sock.close()
                    self.sock = None
                self.queue_message("robot_status",
                    ("Error ❌", '#dc3545', f"Robot connection failed: {e}",
                     tk.NORMAL, tk.DISABLED))
                self.queue_message("error", f"Failed to connect:\n{e}")
                self._add_tcp_log("ERR", f"Connection failed: {e}")

        threading.Thread(target=connect, daemon=True).start()

    def _recv_loop(self):
        while self.is_connected and self.sock:
            try:
                data = self.sock.recv(1024)
                if not data:
                    self._add_tcp_log("INFO", "Connection closed by remote host")
                    break
                decoded = data.decode(errors="ignore")
                self._add_tcp_log("RECV", decoded.strip())
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_connected:
                    self._add_tcp_log("ERR", f"Receive error: {e}")
                break

    def disconnect_robot(self):
        self._add_tcp_log("INFO", "Disconnecting from robot...")
        self.is_connected = False
        if self.sock:
            self.sock.close()
            self.sock = None
        self.robot_status.config(text="Disconnected", fg='#dc3545')
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.status_text.config(text="Robot disconnected")
        self._add_tcp_log("INFO", "TCP connection closed")

    def send_robot_command(self, cmd):
        if not self.is_connected or self.sock is None:
            self.queue_message("status_text", "⚠️ Robot not connected")
            self._add_tcp_log("ERR", f"Cannot send — robot not connected | cmd: {cmd}")
            return False
        try:
            self.sock.sendall((cmd + "\r").encode())
            self._add_tcp_log("SENT", cmd)
            self.queue_message("status_text", f"📤 Sent: {cmd}")
            return True
        except Exception as e:
            self._add_tcp_log("ERR", f"Send failed: {e} | cmd: {cmd}")
            self.queue_message("status_text", f"❌ Command failed: {e}")
            return False

    def update_confidence(self, value):
        self.CONF_THRESHOLD = float(value)
        if self.model:
            self.model.conf = self.CONF_THRESHOLD

    # ──────────────────────────────────────────
    #  Detection & Grid helpers
    # ──────────────────────────────────────────
    def get_class_grid(self, class_name: str) -> dict:
        for key, cfg in CLASS_GRID_CONFIG.items():
            if key.lower() in class_name.lower():
                return cfg
        return DEFAULT_GRID

    def _all_slots(self, cols: int, rows: int):
        """Return semua slot (col, row) dalam urutan baris-per-baris."""
        return [(c + 1, r + 1) for r in range(rows) for c in range(cols)]

    def _next_empty_slot(self, occupied: set, cols: int, rows: int):
        """Return slot kosong pertama yang tidak ada di occupied, atau None."""
        for slot in self._all_slots(cols, rows):
            if slot not in occupied:
                return slot
        return None

    # ──────────────────────────────────────────
    #  Core: detect & build pick-place queue
    # ──────────────────────────────────────────
    def process_frame(self, frame):
        """
        Jalankan deteksi YOLO, gambar grid + lingkaran,
        dan bangun pick_place_queue berdasarkan deteksi.
        """
        if self.model is None:
            return frame

        try:
            results = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            names = results[0].names

            # Reset pallet state setiap kali deteksi ulang
            self.pallet1_occupied.clear()
            self.pallet2_occupied.clear()

            # ── Kumpulkan info per pallet ────────────
            pallet1_info = None   # (grid_cfg, box_coords)
            pallet2_info = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = names.get(cls_id, f"class{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    grid = self.get_class_grid(class_name)

                    if "pick" in class_name.lower():
                        pallet1_info = (grid, x1, y1, x2, y2, class_name)
                    elif "place_1" in class_name.lower():
                        pallet2_info = (grid, x1, y1, x2, y2, class_name)

            # ── Proses Pallet 1 (pick) ───────────────
            if pallet1_info:
                grid, x1, y1, x2, y2, class_name = pallet1_info
                cols, rows = grid["cols"], grid["rows"]
                box_w  = max(1, x2 - x1)
                box_h  = max(1, y2 - y1)
                cell_w = max(1, box_w  // cols)
                cell_h = max(1, box_h  // rows)

                # Gambar grid
                for c in range(1, cols):
                    cv2.line(annotated_frame, (x1 + c*cell_w, y1), (x1 + c*cell_w, y2), (0, 255, 255), 1)
                for r in range(1, rows):
                    cv2.line(annotated_frame, (x1, y1 + r*cell_h), (x2, y1 + r*cell_h), (0, 255, 255), 1)
                cv2.putText(annotated_frame, f"PALLET 1 (PICK) [{cols}×{rows}]",
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                # Deteksi workpiece (lingkaran putih) di pallet 1
                circles = self.detect_white_circles(frame, x1, y1, x2, y2)
                for (cx, cy, r) in circles:
                    gc = min((cx - x1) // cell_w, cols - 1)
                    gr = min((cy - y1) // cell_h, rows - 1)
                    slot = (gc + 1, gr + 1)
                    self.pallet1_occupied.add(slot)

                    label = f"{slot[0]},{slot[1]}"
                    cv2.circle(annotated_frame, (cx, cy), r, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, label, (cx - 20, cy - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    self.detection_count += 1
                    self.last_class_label.config(text=class_name)

                self.detection_label.config(text=str(self.detection_count))

            # ── Proses Pallet 2 (place) ──────────────
            pallet2_grid = None
            if pallet2_info:
                grid, x1p, y1p, x2p, y2p, class_name2 = pallet2_info
                cols2, rows2 = grid["cols"], grid["rows"]
                pallet2_grid = grid
                box_w2  = max(1, x2p - x1p)
                box_h2  = max(1, y2p - y1p)
                cell_w2 = max(1, box_w2  // cols2)
                cell_h2 = max(1, box_h2  // rows2)

                # Gambar grid
                for c in range(1, cols2):
                    cv2.line(annotated_frame, (x1p + c*cell_w2, y1p), (x1p + c*cell_w2, y2p), (0, 255, 100), 1)
                for r in range(1, rows2):
                    cv2.line(annotated_frame, (x1p, y1p + r*cell_h2), (x2p, y1p + r*cell_h2), (0, 255, 100), 1)
                cv2.putText(annotated_frame, f"PALLET 2 (PLACE) [{cols2}×{rows2}]",
                            (x1p, y1p - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

                # Deteksi workpiece yang sudah ada di pallet 2
                circles2 = self.detect_white_circles(frame, x1p, y1p, x2p, y2p)
                for (cx, cy, r) in circles2:
                    gc = min((cx - x1p) // cell_w2, cols2 - 1)
                    gr = min((cy - y1p) // cell_h2, rows2 - 1)
                    slot = (gc + 1, gr + 1)
                    self.pallet2_occupied.add(slot)

                    cv2.circle(annotated_frame, (cx, cy), r, (100, 255, 100), 2)
                    cv2.putText(annotated_frame, f"{slot[0]},{slot[1]}",
                                (cx - 20, cy - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

            # ── Build pick_place_queue ───────────────
            with self.pp_lock:
                self.pick_place_queue.clear()

                if pallet1_info and pallet2_info:
                    grid2_cfg = pallet2_grid
                    cols2 = grid2_cfg["cols"]
                    rows2 = grid2_cfg["rows"]

                    # Salinan occupied pallet 2 untuk simulasi pengisian
                    sim_p2_occupied = set(self.pallet2_occupied)

                    for pick_slot in sorted(self.pallet1_occupied):
                        place_slot = self._next_empty_slot(sim_p2_occupied, cols2, rows2)
                        if place_slot is None:
                            self._add_tcp_log("ERR", "Pallet 2 penuh! Tidak bisa menempatkan semua workpiece.")
                            break
                        self.pick_place_queue.append((pick_slot, place_slot))
                        sim_p2_occupied.add(place_slot)

                    n = len(self.pick_place_queue)
                    self._add_tcp_log("INFO",
                        f"Deteksi selesai: {len(self.pallet1_occupied)} workpiece di Pallet 1 → "
                        f"queue {n} tugas pick & place")
                    self.pp_info_label.config(
                        text=f"{n} workpiece terdeteksi.\nKlik 'Start Sequence' untuk mulai.",
                        fg='#28a745')
                elif pallet1_info:
                    self._add_tcp_log("ERR", "Pallet 2 (place) tidak terdeteksi!")
                    self.pp_info_label.config(text="⚠️ Pallet 2 tidak terdeteksi!", fg='#dc3545')
                elif pallet2_info:
                    self._add_tcp_log("INFO", "Pallet 1 tidak terdeteksi / tidak ada workpiece")
                    self.pp_info_label.config(text="⚠️ Pallet 1 tidak terdeteksi!", fg='#dc3545')
                else:
                    self._add_tcp_log("ERR", "Tidak ada pallet yang terdeteksi!")
                    self.pp_info_label.config(text="⚠️ Tidak ada pallet terdeteksi!", fg='#dc3545')

            # Update monitor tab
            self.queue_message("update_pp_monitor", None)
            return annotated_frame

        except Exception as e:
            self.queue_message("status_text", f"Processing error: {e}")
            self._add_tcp_log("ERR", f"Frame processing error: {e}")
            return frame

    # ──────────────────────────────────────────
    #  Pick & Place Sequence
    # ──────────────────────────────────────────
    def start_pick_place_sequence(self):
        """Mulai sekuens pick & place di thread terpisah."""
        with self.pp_lock:
            if not self.pick_place_queue:
                messagebox.showwarning("Queue Kosong",
                    "Queue kosong!\nLakukan Capture & Detect terlebih dahulu.")
                return
            if self.is_running_sequence:
                messagebox.showinfo("Info", "Sekuens sedang berjalan.")
                return

        self.is_running_sequence = True
        self.start_pp_btn.config(state=tk.DISABLED)
        self.stop_pp_btn.config(state=tk.NORMAL)
        self._add_tcp_log("INFO", "=== PICK & PLACE SEQUENCE DIMULAI ===")
        self.pp_status_label.config(text="▶️ Running...", fg='#28a745')

        threading.Thread(target=self._run_sequence, daemon=True).start()

    def _run_sequence(self):
        """
        Thread: jalankan semua tugas pick & place satu per satu.
        Untuk setiap workpiece di pallet 1:
            1. Jump Pallet(1, col,row)  → pergi ke workpiece
            2. On <output>              → aktifkan gripper (ambil)
            3. Jump Pallet(2, col,row)  → pergi ke slot kosong pallet 2
            4. Off <output>             → nonaktifkan gripper (lepas)
        """
        delay_ms = self.cmd_delay_var.get()
        on_out   = self.on_output_var.get()
        delay_s  = delay_ms / 1000.0

        with self.pp_lock:
            tasks = list(self.pick_place_queue)  # snapshot
        total   = len(tasks)
        done    = 0

        self._pp_log(f"{'─'*60}", "HEADER")
        self._pp_log(f"  Total tugas: {total}", "HEADER")
        self._pp_log(f"  Output gripper: {on_out}", "HEADER")
        self._pp_log(f"  Delay antar perintah: {delay_ms} ms", "HEADER")
        self._pp_log(f"{'─'*60}", "HEADER")

        for idx, (pick_slot, place_slot) in enumerate(tasks):
            if not self.is_running_sequence:
                self._add_tcp_log("INFO", f"Sekuens dihentikan oleh user pada tugas {idx+1}/{total}")
                self._pp_log(f"⏹ Dihentikan pada tugas {idx+1}/{total}", "ERR")
                break

            pick_coord  = f"{pick_slot[0]},{pick_slot[1]}"
            place_coord = f"{place_slot[0]},{place_slot[1]}"

            self._pp_log(f"\n  ── Tugas {idx+1}/{total} ──────────────────────────", "HEADER")
            self._pp_log(f"  🟡 PICK   : Pallet 1 slot ({pick_coord})",  "PICK")
            self._pp_log(f"  🟢 PLACE  : Pallet 2 slot ({place_coord})", "PLACE")

            # ── Step 1: Move to pick position ──
            cmd1 = f"Jump Pallet(1,{pick_coord})"
            self._pp_log(f"       → {cmd1}", "PICK")
            self.send_robot_command(cmd1)
            time.sleep(delay_s)

            if not self.is_running_sequence:
                break

            # ── Step 2: Activate gripper (pick) ──
            cmd2 = f"On {on_out}"
            self._pp_log(f"       → {cmd2}  (gripper ON)", "PICK")
            self.send_robot_command(cmd2)
            time.sleep(delay_s)

            if not self.is_running_sequence:
                break

            # ── Step 3: Move to place position ──
            cmd3 = f"Jump Pallet(2,{place_coord})"
            self._pp_log(f"       → {cmd3}", "PLACE")
            self.send_robot_command(cmd3)
            time.sleep(delay_s)

            if not self.is_running_sequence:
                break

            # ── Step 4: Release gripper (place) ──
            cmd4 = f"Off {on_out}"
            self._pp_log(f"       → {cmd4}  (gripper OFF)", "PLACE")
            self.send_robot_command(cmd4)
            time.sleep(delay_s)

            done += 1
            self._pp_log(f"  ✅ Selesai {done}/{total}", "DONE")

            # Update progress label (safe via after)
            self.root.after(0, self.pp_progress_label.config,
                            {"text": f"{done} / {total} selesai"})
            self.root.after(0, self.pp_queue_label.config,
                            {"text": f"📋 Queue: {total - done} tersisa"})

        # ── Selesai ──
        self.is_running_sequence = False
        self.root.after(0, self.start_pp_btn.config, {"state": tk.NORMAL})
        self.root.after(0, self.stop_pp_btn.config,  {"state": tk.DISABLED})

        if done == total:
            self._pp_log(f"\n{'='*60}", "HEADER")
            self._pp_log(f"  ✅ SEMUA {total} TUGAS SELESAI", "DONE")
            self._pp_log(f"{'='*60}", "HEADER")
            self._add_tcp_log("INFO", f"=== PICK & PLACE SELESAI: {done}/{total} tugas ===")

            # ── Kirim perintah Home ──────────────────
            time.sleep(delay_s)
            self._pp_log(f"\n  🏠 Mengirim perintah HOME...", "HEADER")
            self.send_robot_command("Home")
            self._pp_log(f"       → Go Home", "DONE")
            self._add_tcp_log("INFO", "Robot kembali ke posisi Home")

            self.root.after(0, self.pp_status_label.config,
                            {"text": "🏠 Selesai — Home", "fg": "#28a745"})
        else:
            self.root.after(0, self.pp_status_label.config,
                            {"text": "⏸️ Dihentikan", "fg": "#ffc107"})

    def stop_pick_place_sequence(self):
        """Hentikan sekuens yang sedang berjalan."""
        self.is_running_sequence = False
        self._add_tcp_log("INFO", "Permintaan stop diterima — menghentikan sekuens...")
        self.pp_status_label.config(text="⏸️ Stopping...", fg='#ffc107')

    def reset_pick_place(self):
        """Reset semua state pick & place."""
        if self.is_running_sequence:
            self.stop_pick_place_sequence()
            time.sleep(0.3)
        with self.pp_lock:
            self.pick_place_queue.clear()
        self.pallet1_occupied.clear()
        self.pallet2_occupied.clear()
        self.pp_status_label.config(text="⏸️ Idle", fg='#aaaaaa')
        self.pp_queue_label.config(text="📋 Queue: 0 tasks")
        self.pp_progress_label.config(text="0 / 0 selesai")
        self.pp_info_label.config(
            text="Tekan 'Capture & Detect' dulu,\nlalu 'Start Sequence'", fg='#aaaaaa')
        self.start_pp_btn.config(state=tk.NORMAL)
        self.stop_pp_btn.config(state=tk.DISABLED)
        self.pp_log_text.config(state=tk.NORMAL)
        self.pp_log_text.delete("1.0", tk.END)
        self.pp_log_text.config(state=tk.DISABLED)
        self._add_tcp_log("INFO", "Pick & Place state di-reset")

    # ──────────────────────────────────────────
    #  PP Monitor helpers
    # ──────────────────────────────────────────
    def _pp_log(self, text: str, tag: str = ""):
        """Append text ke pp_log_text (thread-safe via after)."""
        def _do():
            self.pp_log_text.config(state=tk.NORMAL)
            self.pp_log_text.insert(tk.END, text + "\n", tag)
            self.pp_log_text.config(state=tk.DISABLED)
            self.pp_log_text.see(tk.END)
        self.root.after(0, _do)

    def _refresh_pp_monitor(self):
        """Rebuild pp monitor display (called on main thread)."""
        with self.pp_lock:
            tasks = list(self.pick_place_queue)
        total = len(tasks)
        self.pp_queue_label.config(text=f"📋 Queue: {total} tasks")

        self.pp_log_text.config(state=tk.NORMAL)
        self.pp_log_text.delete("1.0", tk.END)
        if not tasks:
            self.pp_log_text.insert(tk.END, "  (belum ada deteksi)\n", "PENDING")
        else:
            self.pp_log_text.insert(tk.END,
                f"  {'#':>3}  {'PICK (Pallet 1)':^18}  {'PLACE (Pallet 2)':^18}\n", "HEADER")
            self.pp_log_text.insert(tk.END, f"  {'─'*50}\n", "HEADER")
            for i, (pick, place) in enumerate(tasks):
                line = f"  {i+1:>3}  ({pick[0]:>2},{pick[1]:>2})  →  ({place[0]:>2},{place[1]:>2})\n"
                self.pp_log_text.insert(tk.END, line, "PENDING")
        self.pp_log_text.config(state=tk.DISABLED)

    # ──────────────────────────────────────────
    #  Circle detection
    # ──────────────────────────────────────────
    def detect_white_circles(self, frame, x1=0, y1=0, x2=None, y2=None):
        if x2 is None: x2 = frame.shape[1]
        if y2 is None: y2 = frame.shape[0]

        roi = frame[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0,   0, 180])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter ** 2) < 0.75:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            detected.append((int(cx) + x1, int(cy) + y1, int(radius)))

        return detected

    # ──────────────────────────────────────────
    #  Capture & UI helpers
    # ──────────────────────────────────────────
    def capture_and_process(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return

        for _ in range(3):
            ret, frame = self.cap.read()

        if ret:
            self._add_tcp_log("INFO", "Frame captured — running YOLO detection...")
            processed_frame = self.process_frame(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((800, 600), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_panel.config(image=imgtk, text="")
            self.camera_panel.image = imgtk
            self.queue_message("status_text", "Frame captured and processed")
        else:
            messagebox.showerror("Error", "Failed to capture frame")
            self._add_tcp_log("ERR", "Failed to capture frame from camera")

    def reset_stats(self):
        self.detection_count = 0
        self.detection_label.config(text="0")
        self.last_class_label.config(text="-")
        self.start_time = time.time()
        self.queue_message("status_text", "Statistics reset")
        self._add_tcp_log("INFO", "Detection statistics reset by user")

    def update_time(self):
        self.time_label.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_time)

    # ──────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────
    def run(self):
        self.start_time = time.time()
        self._add_tcp_log("INFO", "Application started")

        def update_uptime():
            uptime = int(time.time() - self.start_time)
            h, rem = divmod(uptime, 3600)
            m, s = divmod(rem, 60)
            self.uptime_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
            self.root.after(1000, update_uptime)

        update_uptime()

        def on_closing():
            if self.is_running_sequence:
                self.is_running_sequence = False
            if self.is_connected:
                self._add_tcp_log("INFO", "Application closing — disconnecting robot...")
            if self.cap:
                self.cap.release()
            if self.sock:
                self.sock.close()
            self.root.quit()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = ModernYOLOGUI()
    app.run()