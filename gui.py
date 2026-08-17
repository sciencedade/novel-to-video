"""小说转视频 GUI（tkinter 桌面界面）。

运行：python gui.py
提供参数配置、一键生成分镜、一键自动生成视频（含进度与日志显示、停止按钮）。
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

from main import collect_shot_files, load_config, write_json, PROJECT_ROOT
from modules.config_utils import resolve_path

from modules.storyboard_generator import StoryboardGenerator
from modules.continuity_tracker import ContinuityTracker
from modules.comfyui_client import ComfyUIClient, ComfyUIError
from modules.pipeline_runner import PipelineRunner
from modules.video_assembler import VideoAssembler

log = logging.getLogger("novel2video.gui")


class QueueLogHandler(logging.Handler):
    """把日志记录放入线程安全队列，由 Tk 主线程定时取回显示。"""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass


class Novel2VideoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("小说转视频 - ComfyUI MiniMax H3 自动化流水线")
        self.root.geometry("980x760")
        self.root.minsize(860, 640)

        os.chdir(PROJECT_ROOT)
        self._storyboard: Optional[Dict[str, Any]] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # ---------- 变量 ----------
        self.novel_path_var = tk.StringVar(value=str(resolve_path("examples/sample_novel.txt")))
        self.config_path_var = tk.StringVar(value=str(PROJECT_ROOT / "config.yaml"))
        self.mode_var = tk.StringVar(value="auto")
        self.segment_duration_var = tk.DoubleVar(value=10.0)
        self.max_shot_duration_var = tk.DoubleVar(value=10.0)
        self.concurrent_var = tk.IntVar(value=1)
        self.max_retries_var = tk.IntVar(value=3)
        self.auto_run_var = tk.BooleanVar(value=True)
        self.use_ref_var = tk.BooleanVar(value=True)
        self.gen_first_frame_var = tk.BooleanVar(value=False)
        self.skip_storyboard_var = tk.BooleanVar(value=False)

        self.log_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue = queue.Queue()

        self._setup_logging()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._poll_logs)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if not any(isinstance(h, QueueLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(QueueLogHandler(self.log_queue))
        log.info("GUI 日志系统已就绪。")

    def _post_ui(self, fn) -> None:
        """把 UI 更新函数投递到主线程队列（线程安全）。"""
        self.ui_queue.put_nowait(fn)

    def _poll_logs(self) -> None:
        try:
            while True:
                record = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert(tk.END, record + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        try:
            while True:
                fn = self.ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_logs)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # 文件选择
        file_frame = ttk.LabelFrame(self.root, text="输入文件")
        file_frame.pack(fill="x", padx=10, pady=(10, 4))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="小说文本:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Entry(file_frame, textvariable=self.novel_path_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(file_frame, text="浏览…", command=self._browse_novel).grid(row=0, column=2, **pad)

        ttk.Label(file_frame, text="配置文件:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(file_frame, textvariable=self.config_path_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(file_frame, text="浏览…", command=self._browse_config).grid(row=1, column=2, **pad)

        # 分镜参数
        story_frame = ttk.LabelFrame(self.root, text="分镜参数")
        story_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(story_frame, text="分段模式:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Combobox(story_frame, textvariable=self.mode_var, state="readonly",
                     values=["auto", "fixed_duration"], width=14).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(story_frame, text="每镜头时长(秒):").grid(row=0, column=2, sticky="e", **pad)
        ttk.Spinbox(story_frame, from_=2, to=120, increment=1,
                    textvariable=self.segment_duration_var, width=8).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(story_frame, text="单镜头最大时长(秒):").grid(row=0, column=4, sticky="e", **pad)
        ttk.Spinbox(story_frame, from_=2, to=120, increment=1,
                    textvariable=self.max_shot_duration_var, width=8).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(story_frame, text="并发数:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Spinbox(story_frame, from_=1, to=8, increment=1,
                    textvariable=self.concurrent_var, width=8).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(story_frame, text="最大重试:").grid(row=1, column=2, sticky="e", **pad)
        ttk.Spinbox(story_frame, from_=1, to=20, increment=1,
                    textvariable=self.max_retries_var, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Checkbutton(story_frame, text="分镜后立即生成视频 (auto_run)",
                        variable=self.auto_run_var).grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(story_frame, text="使用首帧参考图（上一镜头尾帧）",
                        variable=self.use_ref_var).grid(row=2, column=2, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(story_frame, text="生成首帧图像（ComfyUI 图像工作流）",
                        variable=self.gen_first_frame_var).grid(row=3, column=0, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(story_frame, text="跳过 LLM 分镜，使用已有 storyboard.json",
                        variable=self.skip_storyboard_var).grid(row=3, column=2, columnspan=2, sticky="w", **pad)

        # 操作按钮
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=10, pady=6)
        self.btn_storyboard = ttk.Button(action_frame, text="生成分镜", command=self._on_generate_storyboard)
        self.btn_storyboard.pack(side="left", padx=4)
        self.btn_run = ttk.Button(action_frame, text="一键生成视频", command=self._on_run_full)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(action_frame, text="停止", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_wizard = ttk.Button(action_frame, text="配置向导", command=self._run_wizard)
        self.btn_wizard.pack(side="left", padx=4)
        self.btn_character = ttk.Button(action_frame, text="生成定妆照", command=self._on_generate_character_sheet)
        self.btn_character.pack(side="left", padx=4)
        self.btn_open = ttk.Button(action_frame, text="打开输出目录", command=self._open_output_dir)
        self.btn_open.pack(side="left", padx=4)

        # 进度条
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=10, pady=4)
        self.progress = ttk.Progressbar(progress_frame, maximum=100, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="就绪。请选择小说文件并点击按钮。")
        ttk.Label(progress_frame, textvariable=self.status_var, width=50,
                  anchor="e").pack(side="right", padx=6)

        # 日志区
        log_frame = ttk.LabelFrame(self.root, text="运行日志")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=14, state="disabled",
                                                  font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # 文件浏览
    # ------------------------------------------------------------------
    def _browse_novel(self) -> None:
        path = filedialog.askopenfilename(
            title="选择小说文本",
            initialdir=PROJECT_ROOT,
            filetypes=[("文本文件", "*.txt *.md *.markdown"), ("所有文件", "*.*")],
        )
        if path:
            self.novel_path_var.set(path)

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="选择配置文件",
            initialdir=PROJECT_ROOT,
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if path:
            self.config_path_var.set(path)

    # ------------------------------------------------------------------
    # 配置收集
    # ------------------------------------------------------------------
    def _collect_config(self) -> Dict[str, Any]:
        from modules.config_utils import normalize_config
        cfg = load_config(self.config_path_var.get())
        cfg.setdefault("storyboard", {})["mode"] = self.mode_var.get()
        cfg["storyboard"]["segment_duration_seconds"] = float(self.segment_duration_var.get())
        cfg.setdefault("minimax", {})["max_shot_duration"] = float(self.max_shot_duration_var.get())
        cfg.setdefault("generation", {})["concurrent_jobs"] = int(self.concurrent_var.get())
        cfg["generation"]["max_retries"] = int(self.max_retries_var.get())
        cfg["storyboard"]["use_reference_image"] = bool(self.use_ref_var.get())
        cfg["storyboard"]["generate_first_frame"] = bool(self.gen_first_frame_var.get())
        cfg["generation"]["auto_run"] = bool(self.auto_run_var.get())
        cfg.setdefault("logging", {})["log_dir"] = "logs"
        return normalize_config(cfg)

    # ------------------------------------------------------------------
    # 按钮动作
    # ------------------------------------------------------------------
    def _on_generate_storyboard(self) -> None:
        if self._running:
            return
        self._start_worker(self._run_storyboard_worker)

    def _on_run_full(self) -> None:
        if self._running:
            return
        self._start_worker(self._run_full_worker)

    def _start_worker(self, target) -> None:
        self._stop_event.clear()
        self._set_running(True)
        self.progress["value"] = 0
        self.status_var.set("正在准备…")
        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            self.status_var.set("正在停止：等待当前镜头结束后中断…")
            log.warning("用户请求停止，已发送停止信号。")
        else:
            self.status_var.set("当前没有运行中的任务。")

    def _open_output_dir(self) -> None:
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showinfo("输出目录", f"输出目录：{out_dir}\n（打开失败：{exc}）")

    def _run_wizard(self) -> None:
        """在独立控制台窗口运行首次配置向导。"""
        cli_exe = PROJECT_ROOT / "Novel2Video-CLI.exe"
        if cli_exe.exists():
            cmd = [str(cli_exe), "--wizard"]
        else:
            cmd = [sys.executable, str(PROJECT_ROOT / "wizard.py")]
        try:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), **kwargs)
            self.status_var.set("已启动配置向导（独立窗口）。完成后请重启本程序。")
        except Exception as exc:
            messagebox.showerror("配置向导", f"启动配置向导失败：\n{exc}")

    def _on_generate_character_sheet(self) -> None:
        """打开“用 ComfyUI 生成定妆照”对话框。"""
        if self._running:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("生成角色定妆照（ComfyUI）")
        dialog.geometry("560x260")
        dialog.transient(self.root)
        dialog.grab_set()

        name_var = tk.StringVar()
        prompt_var = tk.StringVar()
        angles_var = tk.StringVar(value="front,side,full")

        ttk.Label(dialog, text="角色名:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
        ttk.Entry(dialog, textvariable=name_var, width=30).grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(dialog, text="提示词(可选):").grid(row=1, column=0, padx=8, pady=8, sticky="e")
        ttk.Entry(dialog, textvariable=prompt_var, width=45).grid(row=1, column=1, padx=8, pady=8)
        ttk.Label(dialog, text="角度(逗号分隔):").grid(row=2, column=0, padx=8, pady=8, sticky="e")
        ttk.Entry(dialog, textvariable=angles_var, width=30).grid(row=2, column=1, padx=8, pady=8)

        def on_ok():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("错误", "角色名不能为空。", parent=dialog)
                return
            angles = [a.strip() for a in angles_var.get().split(",") if a.strip()]
            dialog.destroy()
            self._start_worker(lambda: self._run_character_sheet_worker(
                name, prompt_var.get().strip() or None, angles))

        ttk.Button(dialog, text="生成", command=on_ok).grid(row=3, column=0, columnspan=2, pady=12)
        dialog.bind("<Return>", lambda e: on_ok())

    def _run_character_sheet_worker(self, name: str, prompt: Optional[str],
                                    angles: Optional[list]) -> None:
        try:
            cfg = self._collect_config()
            self._set_status(f"正在生成 {name} 定妆照…")
            from modules.character_sheet_generator import CharacterSheetGenerator
            gen = CharacterSheetGenerator(cfg)
            results = gen.generate(name, prompt=prompt, angles=angles)
            paths = "\n".join(r["path"] for r in results)
            self._set_status("定妆照生成完成")
            self._post_ui(lambda: messagebox.showinfo(
                "完成", f"{name} 定妆照已生成：\n{paths}\n\n"
                        f"请将 reference_image 指向其中一张图。"))
        except Exception as exc:
            log.exception("定妆照生成失败：%s", exc)
            self._set_status("定妆照生成失败")
            self._post_ui(lambda: messagebox.showerror("错误", f"定妆照生成失败：\n{exc}"))
        finally:
            self._set_running(False)

    # ------------------------------------------------------------------
    # 工作线程
    # ------------------------------------------------------------------
    def _run_storyboard_worker(self) -> None:
        try:
            cfg = self._collect_config()
            self._set_status("正在读取小说…")
            novel_path = Path(self.novel_path_var.get())
            if not novel_path.exists():
                novel_path = PROJECT_ROOT / novel_path
            text = novel_path.read_text(encoding="utf-8-sig")
            log.info("读取小说：%s（%d 字符）", novel_path, len(text))

            generator = StoryboardGenerator(cfg)
            storyboard = generator.generate(
                text,
                mode=cfg["storyboard"]["mode"],
                segment_duration=float(cfg["storyboard"]["segment_duration_seconds"]),
            )
            self._storyboard = storyboard
            write_json(PROJECT_ROOT / "storyboard.json", storyboard)
            report = generator.last_report or {}
            report["generated_at"] = datetime.now().isoformat(timespec="seconds")
            report["stage"] = "storyboard"
            write_json(PROJECT_ROOT / "continuity_report.json", report)
            self._set_progress(100)
            self._set_status(f"分镜完成：{storyboard['meta']['total_shots']} 个镜头")
            self._post_ui(lambda: messagebox.showinfo(
                "完成", f"分镜生成完成：共 {storyboard['meta']['total_shots']} 个镜头，"
                        f"总时长 {storyboard['meta']['total_duration_seconds']:.1f} 秒。\n\n"
                        f"已写入 storyboard.json 与 continuity_report.json。"))
        except Exception as exc:
            log.exception("分镜生成失败：%s", exc)
            self._set_status("分镜生成失败")
            self._post_ui(lambda: messagebox.showerror("错误", f"分镜生成失败：\n{exc}"))
        finally:
            self._set_running(False)

    def _run_full_worker(self) -> None:
        try:
            cfg = self._collect_config()
            novel_path = Path(self.novel_path_var.get())
            if not novel_path.exists():
                novel_path = PROJECT_ROOT / novel_path

            # 1) 分镜
            if self.skip_storyboard_var.get():
                storyboard_path = PROJECT_ROOT / "storyboard.json"
                if not storyboard_path.exists():
                    raise FileNotFoundError("勾选了“跳过 LLM 分镜”，但 storyboard.json 不存在。")
                import json as _json
                with open(storyboard_path, "r", encoding="utf-8") as f:
                    storyboard = _json.load(f)
                log.info("使用已有 storyboard.json（%d 个镜头）", len(storyboard.get("shots", [])))
            else:
                self._set_status("正在 LLM 分镜…")
                text = novel_path.read_text(encoding="utf-8-sig")
                log.info("读取小说：%s（%d 字符）", novel_path, len(text))
                generator = StoryboardGenerator(cfg)
                storyboard = generator.generate(
                    text,
                    mode=cfg["storyboard"]["mode"],
                    segment_duration=float(cfg["storyboard"]["segment_duration_seconds"]),
                )
                write_json(PROJECT_ROOT / "storyboard.json", storyboard)
                report = generator.last_report or {}
                report["generated_at"] = datetime.now().isoformat(timespec="seconds")
                report["stage"] = "storyboard"
                write_json(PROJECT_ROOT / "continuity_report.json", report)
                log.info("分镜完成：%d 个镜头", len(storyboard.get("shots", [])))
            self._storyboard = storyboard

            # 2) 生成客户端（ComfyUI 或远程 API）
            provider = cfg.get("_provider", "local_comfyui")
            if provider == "remote_api":
                from modules.remote_api_client import RemoteMiniMaxAPIClient
                self._set_status("正在初始化 MiniMax 远程 API 客户端…")
                client = RemoteMiniMaxAPIClient(cfg)
            else:
                self._set_status("正在连接 ComfyUI 并发现 MiniMax 节点…")
                client = ComfyUIClient(cfg)
                client.build_node_mapping()

            # 3) 顺序生成
            shots_dir = PROJECT_ROOT / cfg.get("generation", {}).get("shots_dir", "shots")
            output_dir = PROJECT_ROOT / cfg.get("generation", {}).get("output_dir", "output")
            shots_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            tracker = ContinuityTracker()
            tracker.initialize_from_storyboard(storyboard)
            runner = PipelineRunner(cfg, storyboard, client, tracker)
            results = runner.run(
                progress_callback=self._on_shot_progress,
                stop_event=self._stop_event,
            )

            # 4) 拼接
            final_path = output_dir / "final.mp4"
            shot_files = collect_shot_files(storyboard, shots_dir)
            if shot_files:
                self._set_status("正在用 ffmpeg 拼接…")
                assembler = VideoAssembler(cfg)
                assembler.assemble(shot_files, storyboard, final_path)
                log.info("最终视频：%s", final_path)
            else:
                log.warning("没有可拼接的镜头文件。")

            # 4.5) 画面层视觉质检（默认开启）
            visual_qa_result = None
            if cfg.get("visual_qa", {}).get("enabled", True) and shot_files:
                from modules.visual_qa import VisualQA
                try:
                    vqa = VisualQA(cfg, storyboard, shots_dir)
                    visual_qa_result = vqa.run()
                    log.info("视觉质检：检查 %d 组，漂移告警 %d，镜像告警 %d",
                             visual_qa_result["summary"]["checked_transitions"],
                             visual_qa_result["summary"]["drift_warnings"],
                             visual_qa_result["summary"]["mirror_warnings"])
                except Exception as exc:
                    log.warning("视觉质检失败（不影响成片）：%s", exc)

            # 5) 报告
            report = tracker.report
            report["generated_at"] = datetime.now().isoformat(timespec="seconds")
            report["stage"] = "generation"
            report["generation_results"] = results
            if visual_qa_result is not None:
                report["visual_qa"] = visual_qa_result
            report["summary"] = {
                "shots_total": len(storyboard.get("shots", []) or []),
                "shots_success": sum(1 for r in results if r.get("status") == "success"),
                "shots_failed": sum(1 for r in results if r.get("status") == "failed"),
                "shots_skipped": sum(1 for r in results if r.get("status") == "skipped"),
                "shots_stopped": sum(1 for r in results if r.get("status") == "stopped"),
                "conflicts": len(tracker.report.get("conflicts", [])),
                "corrections": len(tracker.report.get("corrections", [])),
                "reverse_shots": len(tracker.report.get("reverse_shots", [])),
                "final_video": str(final_path) if final_path.exists() else None,
            }
            if visual_qa_result is not None:
                report["summary"]["visual_drift_warnings"] = visual_qa_result["summary"]["drift_warnings"]
                report["summary"]["visual_mirror_warnings"] = visual_qa_result["summary"]["mirror_warnings"]
            write_json(PROJECT_ROOT / "continuity_report.json", report)

            failed = report["summary"]["shots_failed"]
            self._set_progress(100)
            self._set_status(f"流水线结束：成功 {report['summary']['shots_success']}，失败 {failed}")
            self._post_ui(lambda: messagebox.showinfo(
                "完成" if failed == 0 else "完成（有失败）",
                f"成功 {report['summary']['shots_success']} 个镜头，"
                f"失败 {failed} 个，跳过 {report['summary']['shots_skipped']} 个。\n\n"
                f"最终视频：{final_path if final_path.exists() else '无'}"))
        except ComfyUIError as exc:
            log.error(str(exc))
            self._set_status("ComfyUI 连接/节点错误")
            self._post_ui(lambda: messagebox.showerror(
                "ComfyUI 错误", f"{exc}\n\n请确认 ComfyUI 已启动、MiniMax H3 节点已安装，"
                                f"或检查 config.yaml 的 comfyui.node_mapping。"))
        except Exception as exc:
            log.exception("自动生成失败：%s", exc)
            self._set_status("自动生成失败")
            self._post_ui(lambda: messagebox.showerror("错误", f"自动生成失败：\n{exc}"))
        finally:
            self._set_running(False)

    # ------------------------------------------------------------------
    # 进度与状态
    # ------------------------------------------------------------------
    def _on_shot_progress(self, idx: int, total: int, shot_id: str, status: str) -> None:
        self._post_ui(lambda: self._apply_progress(idx, total, shot_id, status))

    def _apply_progress(self, idx: int, total: int, shot_id: str, status: str) -> None:
        if total > 0:
            self.progress["value"] = min(100.0, idx / total * 100.0)
        self.status_var.set(f"[{idx}/{total}] {shot_id}：{status}")

    def _set_progress(self, value: float) -> None:
        self._post_ui(lambda: self.progress.configure(value=value))

    def _set_status(self, text: str) -> None:
        self._post_ui(lambda: self.status_var.set(text))

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        self._post_ui(lambda: (
            self.btn_storyboard.configure(state=state),
            self.btn_run.configure(state=state),
            self.btn_stop.configure(state="normal" if running else "disabled"),
        ))

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno("退出", "任务正在运行，确定退出吗？"):
                return
            self._stop_event.set()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    Novel2VideoGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
