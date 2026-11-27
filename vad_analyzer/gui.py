"""语音区间 · 声音分析窗口（主要图形入口）。

运行：
    python -m vad_analyzer gui
    bash run_gui.sh

界面把 *原始帧概率/布尔判定* 与 *最终合并区间* 分层画出；阈值等参数改动后
直接用缓存概率重新合并（帧判定与区间合并分离）。活动区间、漏检片段、锁点
冲突与导出核验全部来自真实运算，不使用占位假数据。
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Optional

from ._env import WORKSPACE, ensure_gui_env

ensure_gui_env()

import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, ttk  # noqa: E402

from .audio import AudioPlayer, load_wav  # noqa: E402
from .cuts import load_locked_cuts  # noqa: E402
from .detector import FrameDetector, labels_for_frames  # noqa: E402
from .errors import VadError  # noqa: E402
from .evaluate import frame_confusion  # noqa: E402
from .exporter import export_interval  # noqa: E402
from .kernel import VoiceIntervalKernel  # noqa: E402
from .labels import (find_sidecar, load_annotation,
                     reference_sample_spans)
from .pipeline import config_for

EXAMPLE_WAV = WORKSPACE / "data" / "labeled_short" / "interview_dialogue.wav"
EXAMPLE_CUTS = WORKSPACE / "data" / "labeled_short" / "interview_locked_cuts.json"
DEFAULT_EXPORT_DIR = WORKSPACE / "exports" / "gui"

# Lane geometry (canvas y coordinates).
AXIS_Y0, AXIS_H = 2, 20
WAVE_Y0, WAVE_H = 26, 58
PROB_Y0, PROB_H = 92, 78
RAW_Y0, RAW_H = 178, 15
IV_Y0, IV_H = 201, 24
MISS_Y0, MISS_H = 233, 17
REF_Y0, REF_H = 258, 17
CUT_TOP, CUT_BOTTOM = WAVE_Y0 - 4, REF_Y0 + REF_H + 4
CANVAS_H = 286
LABEL_W = 78

LANE_LABELS = [
    ("时间轴", 12),
    ("波形", WAVE_Y0 + WAVE_H / 2),
    ("帧概率", PROB_Y0 + PROB_H / 2),
    ("原始判定", RAW_Y0 + RAW_H / 2),
    ("活动区间", IV_Y0 + IV_H / 2),
    ("漏检片段", MISS_Y0 + MISS_H / 2),
    ("参考标注", REF_Y0 + REF_H / 2),
]


class VadApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("语音区间 · 声音分析窗口 — vad_analyzer")
        self.root.geometry("1320x880")
        self.root.minsize(1080, 700)

        self.detector = FrameDetector()
        self.cfg = config_for(self.detector)
        self.kernel = VoiceIntervalKernel(self.detector, self.cfg)
        self.player = AudioPlayer()

        self.audio = None
        self.decision = None
        self.annotation = None
        self.annotation_path: Optional[Path] = None
        self.cuts = []
        self.cuts_path: Optional[Path] = None
        self.intervals = []
        self.missed = []
        self.conflicts = []
        self.selected_index: Optional[int] = None
        self.verifications: list[dict] = []

        self.pps = tk.DoubleVar(value=130.0)
        self._wave_cache: tuple[int, np.ndarray, np.ndarray] | None = None
        self._ready = False
        self._auto_exit = tk.BooleanVar(value=True)

        self._build_toolbar()
        self._build_param_bar()
        self._build_status()
        self._build_body()
        self._ready = True
        self._set_param_vars_from_cfg()
        self._log(f"检测器后端：{self.detector.backend}")
        if not self.player.available():
            self._log("播放后端不可用：可导出 WAV 审听；安装 paplay/aplay 或 "
                      "sounddevice 后可直接播放。")

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)
        def btn(text, cmd):
            b = ttk.Button(bar, text=text, command=self._guard(cmd))
            b.pack(side=tk.LEFT, padx=3)
            return b
        btn("打开音频…", self.open_audio)
        btn("示例对白", self.load_example)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        btn("重新分析", self.run_analysis)
        btn("播放选中区间", self.play_selected)
        btn("停止播放", self.stop_play)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        btn("导出选中…", self.export_selected)
        btn("导出全部并核验", self.export_all)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        btn("保存标注草稿…", self.save_annotation_draft)
        btn("载入锁定切点…", self.load_cuts)

        zoom = ttk.Frame(bar)
        zoom.pack(side=tk.RIGHT)
        ttk.Button(zoom, text="−", width=3, command=lambda: self._zoom(0.8)).pack(side=tk.LEFT)
        ttk.Label(zoom, text=" 缩放 ").pack(side=tk.LEFT)
        ttk.Button(zoom, text="+", width=3, command=lambda: self._zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(zoom, text="适应", command=self._zoom_fit).pack(side=tk.LEFT, padx=4)

    def _build_param_bar(self) -> None:
        box = ttk.LabelFrame(self.root, text="参数（帧阈值改动用缓存概率重合并；"
                                             "最短语音按强证据跨度判定）", padding=(8, 4))
        box.pack(side=tk.TOP, fill=tk.X, padx=8)
        self.var_enter = tk.DoubleVar()
        self.var_exit = tk.DoubleVar()
        self.var_min = tk.IntVar()
        self.var_gap = tk.IntVar()
        self.var_pad = tk.IntVar()
        self.var_strong = tk.DoubleVar()
        self._value_labels: dict[str, ttk.Label] = {}

        specs = [
            ("enter", "进入阈值", self.var_enter, 0.15, 0.9, 0.005, "{:.3f}"),
            ("exit", "退出阈值", self.var_exit, 0.05, 0.7, 0.005, "{:.3f}"),
            ("min", "最短语音ms", self.var_min, 40, 400, 10, "{:.0f}"),
            ("gap", "间隙容忍ms", self.var_gap, 100, 1200, 10, "{:.0f}"),
            ("pad", "边界扩展ms", self.var_pad, 0, 250, 5, "{:.0f}"),
            ("strong", "强帧密度≥", self.var_strong, 0.0, 0.8, 0.01, "{:.2f}"),
        ]
        for i, (key, label, var, lo, hi, step, fmt) in enumerate(specs):
            cell = ttk.Frame(box)
            cell.grid(row=0, column=i, padx=8, sticky=tk.W)
            top = ttk.Frame(cell)
            top.pack(fill=tk.X)
            ttk.Label(top, text=label).pack(side=tk.LEFT)
            lv = ttk.Label(top, text="", width=6, foreground="#1a4f8b")
            lv.pack(side=tk.RIGHT)
            self._value_labels[key] = (lv, fmt)
            sc = ttk.Scale(cell, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                           length=132, command=lambda _e, k=key: self._on_param(k))
            sc.pack()
        ttk.Checkbutton(box, text="调进入阈值时联动退出阈值",
                        variable=self._auto_exit).grid(row=0, column=6, padx=8)

    def _build_status(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 2))
        top.pack(side=tk.TOP, fill=tk.X)
        self.summary_var = tk.StringVar(value="尚未载入音频。点击“示例对白”或“打开音频…”。")
        ttk.Label(top, textvariable=self.summary_var, font=("", 10, "bold")).pack(side=tk.LEFT)
        self.error_var = tk.StringVar(value="")
        self.error_label = tk.Label(self.root, textvariable=self.error_var,
                                    fg="#b00020", anchor="w", justify=tk.LEFT,
                                    wraplength=1240)
        self.error_label.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.status_var = tk.StringVar(value="就绪")

    def _build_body(self) -> None:
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        wave_frame = ttk.Frame(paned)
        paned.add(wave_frame, weight=3)

        self.lane_labels = tk.Canvas(wave_frame, width=LABEL_W, height=CANVAS_H,
                                     highlightthickness=0, bg="#f4f4f4")
        self.lane_labels.pack(side=tk.LEFT, fill=tk.Y)
        for text, y in LANE_LABELS:
            self.lane_labels.create_text(6, y, text=text, anchor="w",
                                         fill="#333", font=("", 9))

        mid = ttk.Frame(wave_frame)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(mid, height=CANVAS_H, bg="white",
                                highlightthickness=1, highlightbackground="#ccc")
        xbar = ttk.Scrollbar(mid, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=xbar.set)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        xbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Control-Button-5>", lambda e: self._zoom(0.9))

        nb_frame = ttk.Frame(paned)
        paned.add(nb_frame, weight=2)
        self.nb = ttk.Notebook(nb_frame)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self._build_interval_tab()
        self._build_missed_tab()
        self._build_conflict_tab()
        self._build_verify_tab()
        self._build_log_tab()

        status = ttk.Frame(self.root, relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.status_var, anchor="w",
                  padding=(8, 2)).pack(fill=tk.X)

    def _make_tree(self, parent, columns, widths):
        wrap = ttk.Frame(parent)
        wrap.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(wrap, columns=columns, show="headings", height=8)
        for c, w in zip(columns, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor=tk.W)
        vs = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def _build_interval_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="活动区间")
        cols = ("idx", "start", "end", "dur", "samples", "conf", "low",
                "strong", "level", "gaps")
        widths = (40, 80, 80, 70, 150, 60, 60, 60, 70, 50)
        self.tree_iv = self._make_tree(tab, cols, widths)
        self.tree_iv.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_missed_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="漏检片段")
        self.miss_hint = ttk.Label(tab, foreground="#8a5a00",
                                   text="漏检由“参考语音集合 − 检出区间并集”真实计算。")
        self.miss_hint.pack(anchor="w", padx=6, pady=3)
        cols = ("idx", "start", "end", "dur", "samples")
        self.tree_miss = self._make_tree(tab, cols, (40, 90, 90, 80, 220))

    def _build_conflict_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="锁定切点冲突")
        ttk.Label(tab, foreground="#555",
                  text="锁定切点属于剪辑器：本工具绝不移动锁点，只报告冲突。"
                  ).pack(anchor="w", padx=6, pady=3)
        cols = ("severity", "cut", "time", "kind", "message")
        self.tree_conf = self._make_tree(tab, cols, (80, 110, 80, 220, 520))

    def _build_verify_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="导出核验")
        ttk.Label(tab, foreground="#555",
                  text="导出后重新读取 WAV，逐样本并比对首尾样本。"
                  ).pack(anchor="w", padx=6, pady=3)
        cols = ("file", "n", "first_exp", "first_act", "last_exp", "last_act",
                "all_match", "verified")
        self.tree_ver = self._make_tree(
            tab, cols, (260, 80, 90, 90, 90, 90, 80, 80))

    def _build_log_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="指标 / 日志")
        self.log_text = tk.Text(tab, height=10, wrap=tk.WORD)
        sb = ttk.Scrollbar(tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------- helpers
    def _guard(self, fn):
        def wrapped(*a, **k):
            try:
                self.error_var.set("")
                return fn(*a, **k)
            except VadError as e:
                self._fail(f"{type(e).__name__}: {e}")
            except Exception as e:  # UI boundary: surface everything, never fake ok
                self._fail(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}")
        return wrapped

    def _fail(self, msg: str) -> None:
        first = msg.splitlines()[0]
        self.error_var.set("⚠ " + first)
        self.status_var.set("操作失败：" + first)
        self._log("ERROR " + msg)

    def _log(self, msg: str) -> None:
        if hasattr(self, "log_text"):
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)

    def _set_param_vars_from_cfg(self) -> None:
        c = self.cfg
        self.var_enter.set(c.enter_threshold)
        self.var_exit.set(c.exit_threshold)
        self.var_min.set(c.min_speech_ms)
        self.var_gap.set(c.gap_tolerance_ms)
        self.var_pad.set(c.edge_pad_ms)
        self.var_strong.set(c.min_strong_ratio)
        self._update_value_labels()

    def _update_value_labels(self) -> None:
        vals = {
            "enter": self.var_enter.get(), "exit": self.var_exit.get(),
            "min": self.var_min.get(), "gap": self.var_gap.get(),
            "pad": self.var_pad.get(), "strong": self.var_strong.get(),
        }
        for k, (lab, fmt) in self._value_labels.items():
            lab.configure(text=fmt.format(vals[k]))

    # ------------------------------------------------------------ actions
    def open_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音频",
            filetypes=[("音频/WAV", "*.wav"), ("所有文件", "*.*")])
        if not path:
            return
        self.load_path(Path(path))

    def load_example(self) -> None:
        if not EXAMPLE_WAV.exists():
            from .datasets import generate_labeled_short
            generate_labeled_short(WORKSPACE / "data" / "labeled_short")
        self.load_path(EXAMPLE_WAV, cuts_path=EXAMPLE_CUTS if EXAMPLE_CUTS.exists() else None)

    def load_path(self, path: Path, cuts_path: Optional[Path] = None) -> None:
        self.status_var.set(f"载入中：{path.name}")
        self.root.update_idletasks()
        self.audio = load_wav(path)
        self._wave_cache = None
        self.annotation_path = find_sidecar(path)
        self.annotation = load_annotation(self.annotation_path) if self.annotation_path else None
        self.cuts = []
        self.cuts_path = None
        if cuts_path is not None and Path(cuts_path).exists():
            self.cuts = load_locked_cuts(cuts_path, self.audio.sample_rate)
            self.cuts_path = Path(cuts_path)
        self.selected_index = None
        self._zoom_fit()
        self.run_analysis()
        self.status_var.set(f"已载入 {path}  ({self.audio.duration:.2f}s, "
                            f"{self.audio.sample_rate} Hz)")

    def run_analysis(self) -> None:
        if self.audio is None:
            return
        self.status_var.set("本地模型分析中…")
        self.root.update_idletasks()
        self.decision = self.kernel.detect(self.audio)
        self._log(f"分析完成：{self.audio.path.name if self.audio.path else 'audio'} "
                  f"backend={self.decision.backend} frames={len(self.decision.probs)}")
        self.recompute()

    def _read_config(self):
        if self._auto_exit.get():
            ent = float(self.var_enter.get())
            self.var_exit.set(round(max(0.05, ent - 0.16), 3))
        self._update_value_labels()
        return config_for(
            self.detector,
            enter_threshold=round(float(self.var_enter.get()), 4),
            exit_threshold=round(float(self.var_exit.get()), 4),
            min_speech_ms=int(self.var_min.get()),
            gap_tolerance_ms=int(self.var_gap.get()),
            edge_pad_ms=int(self.var_pad.get()),
            min_strong_ratio=round(float(self.var_strong.get()), 3))

    def _on_param(self, key: str) -> None:
        if not self._ready or self.audio is None:
            self._update_value_labels()
            return
        if key == "enter" and self._auto_exit.get():
            ent = float(self.var_enter.get())
            self.var_exit.set(round(max(0.05, ent - 0.16), 3))
        self._update_value_labels()
        self.recompute()

    def recompute(self) -> None:
        if self.decision is None:
            return
        self.cfg = self._read_config()
        result = self.kernel.recompute(
            self.audio, self.decision, self.cfg,
            self.annotation, self.cuts)
        self.intervals = result.intervals
        self.missed = result.missed
        self.conflicts = result.conflicts
        self._refresh_interval_tree()
        self._refresh_miss_tree()
        self._refresh_conflict_tree()
        self._refresh_summary()
        self._refresh_metrics()
        self._draw()

    def _refresh_summary(self) -> None:
        speech = sum(iv.end_sample - iv.start_sample for iv in self.intervals)
        ratio = speech / max(1, len(self.audio.samples))
        ann = "有标注" if self.annotation else "无同名标注（漏检不可计算）"
        self.summary_var.set(
            f"{len(self.intervals)} 个活动区间 · 语音活动 {speech/self.audio.sample_rate:.2f}s "
            f"({ratio*100:.1f}%) · 漏检 {len(self.missed)} · 锁点冲突 "
            f"{len(self.conflicts)} · {ann} · 播放后端 {self.player.backend}")

    def _refresh_metrics(self) -> None:
        if self.annotation is None or self.decision is None:
            return
        try:
            exact = self.annotation.get("active_spans_samples")
            from .labels import active_sample_spans
            exact_spans = active_sample_spans(self.annotation, self.audio.sample_rate) \
                or reference_sample_spans(self.annotation, self.audio.sample_rate)
            y_true = labels_for_frames(self.decision.features.grid, exact_spans,
                                       len(self.audio.samples))
            fc = frame_confusion(y_true, self.decision.raw)
            self._log(
                "帧指标: " + ", ".join(
                    f"{k}={v}" for k, v in fc.items()
                    if k in ("accuracy", "speech_frame_miss_rate",
                             "nonspeech_false_positive_rate", "silence_frame_ratio"))
                + "  (accuracy 受长静音抬高，勿只看它)")
        except Exception as e:  # metrics are auxiliary
            self._log("指标计算失败: " + repr(e))

    # --------------------------------------------------------- tree refresh
    def _clear_tree(self, tree) -> None:
        for iid in tree.get_children():
            tree.delete(iid)

    def _refresh_interval_tree(self) -> None:
        t = self.tree_iv
        self._clear_tree(t)
        for iv in self.intervals:
            samples = f"{iv.start_sample}:{iv.end_sample}"
            t.insert("", tk.END, iid=str(iv.index), values=(
                iv.index, f"{iv.start:.3f}", f"{iv.end:.3f}",
                f"{iv.duration:.2f}", samples, f"{iv.confidence:.2f}",
                f"{iv.confidence_low:.2f}", f"{iv.strong_frame_ratio:.2f}",
                iv.confidence_level, iv.bridged_gaps))
        if self.selected_index is not None:
            iid = str(self.selected_index)
            if t.exists(iid):
                t.selection_set(iid)

    def _refresh_miss_tree(self) -> None:
        t = self.tree_miss
        self._clear_tree(t)
        if self.annotation is None:
            self.miss_hint.configure(
                text="未找到同名 .json 参考标注：漏检片段需要真实标注才能计算，"
                     "当前不显示任何漏检（不造假）。")
        else:
            self.miss_hint.configure(
                text=f"参考语音 − 检出区间并集，min_ms=120；共 {len(self.missed)} 段。")
        for i, m in enumerate(self.missed, 1):
            t.insert("", tk.END, values=(
                i, f"{m.start:.3f}", f"{m.end:.3f}", f"{m.duration:.2f}",
                f"{m.start_sample}:{m.end_sample}"))

    def _refresh_conflict_tree(self) -> None:
        t = self.tree_conf
        self._clear_tree(t)
        sev_color = {"error": "#b00020", "warning": "#9a6a00", "info": "#1a4f8b"}
        for i, c in enumerate(self.conflicts, 1):
            iid = f"cf{i}"
            t.insert("", tk.END, iid=iid, values=(
                c.severity, c.cut_id, f"{c.cut_time_sec:.3f}", c.kind, c.message))
            t.tag_configure(c.severity, foreground=sev_color.get(c.severity, "#000"))
            t.item(iid, tags=(c.severity,))
        if not self.cuts:
            self._clear_tree(t)

    def _add_verification(self, path: Path, v: dict) -> None:
        self.verifications.append(v)
        self.tree_ver.insert("", tk.END, values=(
            Path(v["path"]).name, v["expected_sample_count"],
            v["first_sample_expected"], v["first_sample_actual"],
            v["last_sample_expected"], v["last_sample_actual"],
            v["all_samples_match"], v["verified"]))

    # ----------------------------------------------------------- selection
    def _on_tree_select(self, _e=None) -> None:
        sel = self.tree_iv.selection()
        if not sel:
            return
        try:
            self.selected_index = int(sel[0])
        except ValueError:
            return
        self._draw()

    def _interval_at_time(self, t_sec: float):
        s = int(round(t_sec * self.audio.sample_rate))
        for iv in self.intervals:
            if iv.start_sample <= s < iv.end_sample:
                return iv
        return None

    def _on_canvas_click(self, event) -> None:
        if self.audio is None:
            return
        x = self.canvas.canvasx(event.x)
        t_sec = x / float(self.pps.get())
        iv = self._interval_at_time(max(0.0, t_sec))
        if iv is not None:
            self.selected_index = iv.index
            if self.tree_iv.exists(str(iv.index)):
                self.tree_iv.selection_set(str(iv.index))
                self.tree_iv.see(str(iv.index))
        else:
            self.selected_index = None
            self.tree_iv.selection_remove(self.tree_iv.selection())
        self._draw()

    def _on_canvas_double(self, event) -> None:
        if self.audio is None:
            return
        x = self.canvas.canvasx(event.x)
        iv = self._interval_at_time(x / float(self.pps.get()))
        if iv is not None:
            self.selected_index = iv.index
            self.play_interval(iv)

    # -------------------------------------------------------------- zoom
    def _zoom(self, factor: float) -> None:
        if self.audio is None:
            return
        new_pps = float(np.clip(self.pps.get() * factor, 24.0, 900.0))
        self.pps.set(new_pps)
        self._wave_cache = None
        self._draw()

    def _zoom_fit(self) -> None:
        if self.audio is None:
            return
        try:
            avail = max(300, self.canvas.winfo_width() - 8)
        except Exception:
            avail = 1100
        self.pps.set(max(30.0, min(220.0, avail / max(0.1, self.audio.duration))))
        self._wave_cache = None
        self._draw()

    def _on_ctrl_wheel(self, event) -> None:
        self._zoom(1.1 if event.delta > 0 else 0.9)

    # ------------------------------------------------------------ playback
    def play_selected(self) -> None:
        if self.audio is None:
            self._fail("请先打开或示例载入音频。")
            return
        iv = self._selected_interval()
        if iv is None:
            self._fail("请先在列表或画布中选择一个活动区间。")
            return
        self.play_interval(iv)

    def _selected_interval(self):
        if self.selected_index is None:
            return None
        for iv in self.intervals:
            if iv.index == self.selected_index:
                return iv
        return None

    def play_interval(self, iv) -> None:
        if not self.player.available():
            self._fail(f"没有可用播放后端（检测为 {self.player.backend}）。"
                       "可先“导出”WAV 审听，或安装 paplay/aplay/sounddevice。")
            return
        try:
            self.status_var.set(f"播放区间 #{iv.index} …")
            self.player.play_buffer(self.audio, iv.start_sample, iv.end_sample)
            self.status_var.set(f"已发送播放：区间 #{iv.index} "
                                f"{iv.start:.2f}-{iv.end:.2f}s")
        except Exception as e:
            self._fail(f"播放失败：{type(e).__name__}: {e}")

    def stop_play(self) -> None:
        try:
            self.player.stop()
            self.status_var.set("已停止播放")
        except Exception as e:
            self._fail(f"停止播放失败：{e}")

    # ------------------------------------------------------------ export
    def export_selected(self) -> None:
        iv = self._selected_interval()
        if iv is None:
            self._fail("请先选择要导出的活动区间。")
            return
        d = filedialog.askdirectory(title="选择导出目录",
                                    initialdir=str(DEFAULT_EXPORT_DIR))
        if not d:
            return
        self._export_all(Path(d), only_selected=True)

    def export_all(self) -> None:
        if not self.intervals:
            self._fail("当前没有可导出的活动区间。")
            return
        d = filedialog.askdirectory(title="选择导出目录",
                                    initialdir=str(DEFAULT_EXPORT_DIR))
        if not d:
            return
        self._export_all(Path(d), only_selected=False)

    def _export_all(self, out_dir: Path, only_selected: bool) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [self._selected_interval()] if only_selected else self.intervals
        targets = [iv for iv in targets if iv is not None]
        ok = 0
        for iv in targets:
            path, ver = export_interval(self.audio, iv, out_dir)
            self._add_verification(path, ver.to_dict())
            if ver.verified:
                ok += 1
            self._log(f"导出 {path.name} verified={ver.verified} "
                      f"first {ver.first_sample_expected}=={ver.first_sample_actual} "
                      f"last {ver.last_sample_expected}=={ver.last_sample_actual}")
        self.status_var.set(f"导出完成：{ok}/{len(targets)} 个区间逐样本核验通过 → {out_dir}")

    def save_annotation_draft(self) -> None:
        """Write current intervals as a reviewable annotation draft.

        This is deliberately labelled as a VAD draft, not a human annotation:
        a reviewer opens it, corrects start/end, and re-saves as the reference.
        """
        if self.audio is None or not self.intervals:
            self._fail("没有可保存的区间，请先载入并分析音频。")
            return
        initial = "annotation_draft.json"
        if self.audio.path is not None:
            initial = self.audio.path.with_suffix(".draft.json").name
        path = filedialog.asksaveasfilename(
            title="保存标注草稿（需人工复核）",
            initialdir=str(self.audio.path.parent if self.audio.path else WORKSPACE),
            initialfile=initial,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        import json
        segments = [{
            "start": round(iv.start, 4), "end": round(iv.end, 4),
            "core_start": round(iv.core_start, 4),
            "core_end": round(iv.core_end, 4),
            "kind": "speech", "label": "voice_activity_draft",
            "confidence": round(iv.confidence, 3),
        } for iv in self.intervals]
        doc = {
            "audio_file": self.audio.path.name if self.audio.path else "",
            "sample_rate": self.audio.sample_rate,
            "duration_sec": round(self.audio.duration, 4),
            "annotator": "vad-draft-needs-human-review",
            "note": "由 VAD 区间生成的草稿；请人工核对/修正 start/end 后作为参考标注。",
            "segments": segments,
        }
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        self._log(f"标注草稿已保存：{path}（{len(segments)} 段，待人工复核）")
        self.status_var.set(f"标注草稿已保存：{path}")

    def load_cuts(self) -> None:
        if self.audio is None:
            self._fail("请先载入音频，再载入匹配采样率的锁定切点。")
            return
        path = filedialog.askopenfilename(
            title="选择锁定切点 JSON",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.cuts = load_locked_cuts(path, self.audio.sample_rate)
            self.cuts_path = Path(path)
            self.recompute()
            self.status_var.set(f"已载入 {len(self.cuts)} 个锁定切点（不会移动）")
        except Exception as e:
            self._fail(f"锁定切点载入失败：{type(e).__name__}: {e}")

    # -------------------------------------------------------------- draw
    def _lane(self, y0, h, fill=None, outline="#e2e2e2"):
        if fill:
            self.canvas.create_rectangle(0, y0, 10 ** 9, y0 + h, fill=fill,
                                         outline=outline)

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.audio is None:
            c.create_text(300, 120, text="载入音频后在此显示波形 / 帧概率 / 区间",
                          fill="#888", font=("", 13))
            c.configure(scrollregion=(0, 0, 640, CANVAS_H))
            return
        pps = float(self.pps.get())
        dur = self.audio.duration
        width = int(dur * pps) + 24
        c.configure(scrollregion=(0, 0, width, CANVAS_H))

        for y0, h, fill in [
            (WAVE_Y0, WAVE_H, "#fafafa"), (PROB_Y0, PROB_H, "#f6f8fb"),
            (RAW_Y0, RAW_H, "#fafafa"), (IV_Y0, IV_H, "#f3f8f3"),
            (MISS_Y0, MISS_H, "#fdf3f3"), (REF_Y0, REF_H, "#f3f6fd")]:
            c.create_rectangle(0, y0, width, y0 + h, fill=fill, outline="#e6e6e6")

        self._draw_axis(width, pps)
        self._draw_waveform(pps, width)
        self._draw_probability(pps)
        self._draw_raw(pps)
        self._draw_refs(pps)
        self._draw_missed(pps)
        self._draw_intervals(pps)
        self._draw_cuts(pps)

    def _draw_axis(self, width, pps) -> None:
        c = self.canvas
        if pps >= 120:
            step = 0.5
        elif pps >= 55:
            step = 1.0
        else:
            step = 2.0
        t = 0.0
        while t <= self.audio.duration + 1e-6:
            x = t * pps
            c.create_line(x, AXIS_Y0 + AXIS_H - 6, x, AXIS_Y0 + AXIS_H, fill="#999")
            c.create_text(x + 3, AXIS_Y0 + 4, text=f"{t:.1f}s", anchor="nw",
                          fill="#555", font=("", 8))
            t += step

    def _wave_peaks(self, ncol: int):
        if self._wave_cache is not None and self._wave_cache[0] == ncol:
            return self._wave_cache[1], self._wave_cache[2]
        x = self.audio.float
        n = len(x)
        edges = np.linspace(0, n, ncol + 1).astype(np.int64)
        mn = np.zeros(ncol, dtype=np.float32)
        mx = np.zeros(ncol, dtype=np.float32)
        for i in range(ncol):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                b = min(n, a + 1)
            seg = x[a:b]
            if seg.size:
                mn[i] = float(seg.min())
                mx[i] = float(seg.max())
        self._wave_cache = (ncol, mn, mx)
        return mn, mx

    def _draw_waveform(self, pps, width) -> None:
        c = self.canvas
        ncol = max(1, int(self.audio.duration * pps))
        mn, mx = self._wave_peaks(ncol)
        mid = WAVE_Y0 + WAVE_H / 2
        amp = WAVE_H / 2 - 3
        top_pts, bot_pts = [], []
        for i in range(ncol):
            x = i
            top_pts.extend((x, mid - float(mx[i]) * amp))
            bot_pts.append((x, mid - float(mn[i]) * amp))
        pts = top_pts + [v for xy in reversed(bot_pts) for v in xy]
        c.create_polygon(pts, fill="#9db8d2", outline="")
        c.create_line(0, mid, width, mid, fill="#d0d0d0")

    def _draw_probability(self, pps) -> None:
        c = self.canvas
        if self.decision is None:
            return
        probs = np.asarray(self.decision.probs)
        starts = np.asarray(self.decision.start_samples) / self.audio.sample_rate
        y0, h = PROB_Y0, PROB_H
        base = y0 + h - 4

        def yv(p):
            return base - float(np.clip(p, 0, 1)) * (h - 8)

        pts = []
        for t, p in zip(starts, probs):
            pts.extend((t * pps, yv(p)))
        if len(pts) >= 4:
            c.create_line(*pts, fill="#1a4f8b", width=1)
        ent = self.cfg.enter_threshold
        ex = self.cfg.exit_threshold
        for val, color, tag in [(ent, "#d35400", "进入"), (ex, "#e0a030", "退出")]:
            y = yv(val)
            c.create_line(0, y, 10 ** 9, y, fill=color, dash=(5, 3))
            c.create_text(6, y - 7, text=f"{tag} {val:.2f}", anchor="w",
                          fill=color, font=("", 8))
        c.create_line(0, base, 10 ** 9, base, fill="#ccc")

    def _draw_raw(self, pps) -> None:
        c = self.canvas
        if self.decision is None:
            return
        raw = np.asarray(self.decision.raw, dtype=bool)
        starts = np.asarray(self.decision.start_samples) / self.audio.sample_rate
        hop_w = max(1.0, (self.decision.features.grid.hop
                          / self.audio.sample_rate) * pps)
        i = 0
        n = len(raw)
        while i < n:
            if not raw[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and raw[j + 1]:
                j += 1
            x0 = starts[i] * pps
            x1 = (starts[j] * pps) + hop_w
            c.create_rectangle(x0, RAW_Y0 + 2, x1, RAW_Y0 + RAW_H - 2,
                               fill="#2e7d32", outline="")
            i = j + 1

    def _draw_intervals(self, pps) -> None:
        c = self.canvas
        for iv in self.intervals:
            x0, x1 = iv.start * pps, iv.end * pps
            core0, core1 = iv.core_start * pps, iv.core_end * pps
            sel = iv.index == self.selected_index
            fill = "#7fb3ff" if sel else "#b9d6f2"
            outline = "#0b3d91" if sel else "#4a78a8"
            c.create_rectangle(x0, IV_Y0 + 3, x1, IV_Y0 + IV_H - 3,
                               fill=fill, outline=outline, width=2 if sel else 1)
            c.create_rectangle(core0, IV_Y0 + 6, core1, IV_Y0 + IV_H - 6,
                               fill="#3f7fc4", outline="")
            if x1 - x0 > 34:
                c.create_text(x0 + 5, IV_Y0 + IV_H / 2,
                              text=f"#{iv.index} {iv.confidence:.2f}",
                              anchor="w", fill="#08264a", font=("", 8, "bold"))

    def _draw_missed(self, pps) -> None:
        c = self.canvas
        for m in self.missed:
            x0, x1 = m.start * pps, m.end * pps
            c.create_rectangle(x0, MISS_Y0 + 2, x1, MISS_Y0 + MISS_H - 2,
                               fill="#e57373", outline="#b71c1c")
            if x1 - x0 > 30:
                c.create_text((x0 + x1) / 2, MISS_Y0 + MISS_H / 2,
                              text=f"{m.duration:.2f}s", fill="white",
                              font=("", 7))

    def _draw_refs(self, pps) -> None:
        if self.annotation is None:
            return
        c = self.canvas
        refs = reference_sample_spans(self.annotation, self.audio.sample_rate)
        for a, b in refs:
            x0, x1 = a / self.audio.sample_rate * pps, b / self.audio.sample_rate * pps
            c.create_rectangle(x0, REF_Y0 + 2, x1, REF_Y0 + REF_H - 2,
                               outline="#2b579a", width=1.5)

    def _draw_cuts(self, pps) -> None:
        c = self.canvas
        sev = {cf.cut_id: cf.severity for cf in self.conflicts}
        for k, cut in enumerate(self.cuts):
            x = cut.time_sec * pps
            color = {"error": "#b00020", "warning": "#d08a00",
                     "info": "#1a4f8b"}.get(sev.get(cut.cut_id), "#7b1fa2")
            c.create_line(x, CUT_TOP, x, CUT_BOTTOM, fill=color,
                          dash=(6, 3), width=2)
            c.create_polygon(x - 5, CUT_TOP, x + 5, CUT_TOP, x, CUT_TOP + 7,
                             fill=color)
            c.create_text(x + 3, CUT_BOTTOM + 2, text=cut.cut_id, anchor="nw",
                          fill=color, font=("", 7))


def main(argv: Optional[list[str]] = None) -> int:
    ensure_gui_env()
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = VadApp(root)
    quit_ms = os.environ.get("VAD_GUI_QUIT_MS")
    if quit_ms:
        root.after(int(quit_ms), root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
