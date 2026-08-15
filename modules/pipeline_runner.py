"""一键顺序不间断自动生成调度器。

按 shot_id 顺序向 ComfyUI 提交镜头任务：
- 顺序执行（默认）保证空间衔接：上一镜头完成后立即下载视频并提交下一镜头
- 失败自动重试，重试时由 ContinuityTracker 生成空间锚点修正提示词
- 断点续传：跳过 shots/ 目录中已成功生成的镜头
- 可选并发提交（concurrent_jobs > 1），提示词仍按顺序生成
"""

from __future__ import annotations

import logging
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comfyui_client import ComfyUIClient
from .continuity_tracker import ContinuityTracker

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class PipelineRunner:
    """顺序视频生成调度器。"""

    def __init__(
        self,
        config: Dict[str, Any],
        storyboard: Dict[str, Any],
        client: ComfyUIClient,
        tracker: ContinuityTracker,
    ):
        self.config = config
        self.storyboard = storyboard
        self.client = client
        self.tracker = tracker
        gen_cfg = config.get("generation", {}) or {}
        self.concurrent_jobs = int(gen_cfg.get("concurrent_jobs", 1))
        self.max_retries = int(gen_cfg.get("max_retries", 3))
        self.retry_backoff = float(gen_cfg.get("retry_backoff_seconds", 5))
        self.output_dir = Path(gen_cfg.get("output_dir", "output"))
        self.shots_dir = Path(gen_cfg.get("shots_dir", "shots"))
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        storyboard_cfg = config.get("storyboard", {}) or {}
        self.use_reference_image = bool(storyboard_cfg.get("use_reference_image", True))
        self.generate_first_frame = bool(storyboard_cfg.get("generate_first_frame", False))
        self.first_frame_prompt = str(storyboard_cfg.get("first_frame_prompt", ""))
        minimax_cfg = config.get("minimax", {}) or {}
        self.seed = int(minimax_cfg.get("seed", -1))
        from .config_utils import detect_ffmpeg
        self.ffmpeg = detect_ffmpeg(config)
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, progress_callback=None, stop_event=None) -> List[Dict[str, Any]]:
        """执行调度。

        progress_callback: 可选，签名 (index, total, shot_id, status)
        stop_event: 可选 threading.Event，设置后将在安全点中断生成
        """
        self._progress_callback = progress_callback
        self._stop_event = stop_event
        shots = self.storyboard.get("shots", []) or []
        if not shots:
            raise ValueError("storyboard 中没有镜头，无法运行。")

        todo: List[Dict[str, Any]] = []
        last_video: Optional[str] = None
        last_shot: Optional[Dict[str, Any]] = None
        total = len(shots)
        for idx, shot in enumerate(shots, 1):
            if self._stop_event and self._stop_event.is_set():
                log.info("收到停止信号，终止调度。")
                self.results.append({"shot_id": shot.get("shot_id"), "status": "stopped"})
                continue
            if self._is_done(shot):
                # 断点续传：已完成的镜头也要推进空间状态机
                self.tracker.prepare_shot(shot)
                self.tracker.apply_shot(shot)
                existing = self._existing_output(shot)
                if existing:
                    last_video = existing
                    last_shot = shot
                log.info("跳过已完成镜头 %s", shot["shot_id"])
                self.results.append({"shot_id": shot["shot_id"], "status": "skipped",
                                     "video_path": existing})
            else:
                todo.append(shot)
            if self._progress_callback:
                self._progress_callback(idx, total, shot.get("shot_id"),
                                        "skipped" if self._is_done(shot) else "pending")

        done = total - len(todo)
        log.info("共 %d 个镜头，已完成 %d 个，待生成 %d 个。并发=%d",
                 total, done, len(todo), self.concurrent_jobs)
        if not todo:
            log.info("所有镜头均已生成，无需提交。")
            return self.results

        if self.concurrent_jobs > 1:
            self._run_concurrent(todo, total, done)
        else:
            self._run_sequential(todo, total, done, initial_prev_video=last_video,
                                 initial_prev_shot=last_shot)

        ok = sum(1 for r in self.results if r.get("status") == "success")
        failed = sum(1 for r in self.results if r.get("status") == "failed")
        stopped = sum(1 for r in self.results if r.get("status") == "stopped")
        log.info("调度完成：成功 %d，失败 %d，跳过 %d，停止 %d", ok, failed, done, stopped)
        return self.results

    # ------------------------------------------------------------------
    # 顺序 / 并发
    # ------------------------------------------------------------------
    def _run_sequential(
        self,
        todo: List[Dict[str, Any]],
        total: int,
        done: int,
        initial_prev_video: Optional[str] = None,
        initial_prev_shot: Optional[Dict[str, Any]] = None,
    ) -> None:
        prev_video: Optional[str] = initial_prev_video
        prev_shot: Optional[Dict[str, Any]] = initial_prev_shot
        use_tqdm = tqdm is not None and self._progress_callback is None
        iterator = tqdm(todo, desc="生成镜头", unit="shot") if use_tqdm else todo
        for idx, shot in enumerate(iterator):
            if self._stop_event and self._stop_event.is_set():
                log.info("收到停止信号，中断顺序生成。")
                self.results.append({"shot_id": shot.get("shot_id"), "status": "stopped"})
                break
            next_shot = todo[idx + 1] if idx + 1 < len(todo) else None
            self._prepare_shot(shot, prev_video=prev_video, prev_shot=prev_shot, next_shot=next_shot)
            try:
                result = self._process_shot(shot)
                self.tracker.apply_shot(shot)
                prev_video = result.get("video_path")
                prev_shot = shot
            except Exception as exc:
                result = {"shot_id": shot.get("shot_id"), "status": "failed",
                          "error": str(exc), "attempts": self.max_retries}
                log.error("镜头 %s 最终失败：%s", shot.get("shot_id"), exc)
            self.results.append(result)
            if self._progress_callback:
                self._progress_callback(done + idx + 1, total, shot.get("shot_id"),
                                        result.get("status", "unknown"))
            if use_tqdm:
                iterator.set_postfix(shot=shot.get("shot_id"),
                                     status=result.get("status", "unknown"))

    def _run_concurrent(self, todo: List[Dict[str, Any]], total: int, done: int) -> None:
        log.warning("并发模式（concurrent_jobs=%d）：提示词按顺序生成，但参考图链路"
                    "退化为 storyboard 静态参考图，空间衔接以文本锚点为准。", self.concurrent_jobs)
        # 提示词必须按顺序准备：先顺序跑一遍 tracker
        for shot in todo:
            if self._stop_event and self._stop_event.is_set():
                log.info("收到停止信号，中断并发生成。")
                self.results.append({"shot_id": shot.get("shot_id"), "status": "stopped"})
                return
            self._prepare_shot(shot, prev_video=None, prev_shot=None, next_shot=None)
            self.tracker.apply_shot(shot)

        use_tqdm = tqdm is not None and self._progress_callback is None
        with ThreadPoolExecutor(max_workers=self.concurrent_jobs) as executor:
            futures = {shot.get("shot_id"): executor.submit(self._process_shot, shot)
                       for shot in todo}
            iterator = tqdm(futures.items(), desc="收集镜头", unit="shot") if use_tqdm else futures.items()
            for idx, (shot_id, future) in enumerate(iterator, 1):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"shot_id": shot_id, "status": "failed",
                              "error": str(exc), "attempts": self.max_retries}
                    log.error("镜头 %s 最终失败：%s", shot_id, exc)
                self.results.append(result)
                if self._progress_callback:
                    self._progress_callback(done + idx, total, shot_id,
                                            result.get("status", "unknown"))

    # ------------------------------------------------------------------
    # 单镜头准备与执行
    # ------------------------------------------------------------------
    def _prepare_shot(
        self,
        shot: Dict[str, Any],
        prev_video: Optional[str],
        prev_shot: Optional[Dict[str, Any]],
        next_shot: Optional[Dict[str, Any]],
    ) -> None:
        """生成前准备：推进空间状态机、确定首尾帧参考图。"""
        self.tracker.prepare_shot(shot)
        shot_id = shot.get("shot_id", "shot_000")

        start_image: Optional[str] = None
        end_image: Optional[str] = None

        if self.use_reference_image:
            # 新场景或首镜头：优先用户提供的参考图，其次生成首帧图
            if prev_video and prev_shot and prev_shot.get("scene_id") == shot.get("scene_id"):
                start_image = self._extract_last_frame(prev_video, prev_shot.get("shot_id", "prev"))
            if not start_image:
                start_image = shot.get("reference_image") or None
            if not start_image and prev_video is None and self.generate_first_frame:
                first_prompt = str(shot.get("start_frame_prompt") or shot.get("action") or "")
                if self.first_frame_prompt:
                    first_prompt = f"{self.first_frame_prompt}, {first_prompt}"
                generated = self.client.generate_first_frame(first_prompt, shot_id)
                if generated:
                    start_image = generated
                    shot["reference_image"] = generated

            # 尾帧参考图：优先本镜头显式指定，其次取下一镜头首帧参考图（同场景）
            end_image = shot.get("end_frame_image") or None
            if not end_image and next_shot and next_shot.get("reference_image") \
                    and next_shot.get("scene_id") == shot.get("scene_id"):
                end_image = next_shot.get("reference_image")

        shot["_start_image"] = start_image
        shot["_end_image"] = end_image
        log.debug("镜头 %s 首帧参考=%s 尾帧参考=%s", shot_id, start_image, end_image)

    def _process_shot(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """提交并等待单个镜头，失败自动重试。"""
        shot_id = shot.get("shot_id", "shot_000")
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            if self._stop_event and self._stop_event.is_set():
                log.info("[%s] 收到停止信号，中断重试。", shot_id)
                return {"shot_id": shot_id, "status": "stopped",
                        "error": "用户停止", "attempts": attempt}
            try:
                prompt = str(shot.get("video_prompt") or "")
                if attempt > 1:
                    prompt = f"{prompt}\n{self.tracker.retry_adjustment(shot, attempt, last_error)}"
                seed = self.seed if self.seed >= 0 else random.randint(0, 2**31 - 1)
                workflow = self.client.build_workflow(
                    shot,
                    prompt=prompt,
                    start_image=shot.get("_start_image"),
                    end_image=shot.get("_end_image"),
                    seed=seed,
                )
                prompt_id = self.client.submit_workflow(workflow)
                log.info("[%s] 第 %d/%d 次尝试，prompt_id=%s seed=%s",
                         shot_id, attempt, self.max_retries, prompt_id, seed)
                entry = self.client.wait_for_completion(prompt_id)
                video_path = self.client.download_output(entry, self.shots_dir, shot_id)
                if not video_path or not Path(video_path).exists():
                    raise RuntimeError("ComfyUI 历史记录中没有可下载的视频输出")
                log.info("[%s] 生成成功：%s", shot_id, video_path)
                return {"shot_id": shot_id, "status": "success",
                        "video_path": str(video_path), "attempts": attempt,
                        "prompt_id": prompt_id}
            except Exception as exc:
                last_error = str(exc)
                log.warning("[%s] 第 %d 次尝试失败：%s", shot_id, attempt, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        return {"shot_id": shot_id, "status": "failed",
                "error": last_error, "attempts": self.max_retries}

    # ------------------------------------------------------------------
    # 断点检测与工具
    # ------------------------------------------------------------------
    def _is_done(self, shot: Dict[str, Any]) -> bool:
        path = self._existing_output(shot)
        if not path:
            return False
        try:
            return Path(path).stat().st_size > 10_000  # 大于 10KB 视为有效视频
        except OSError:
            return False

    def _existing_output(self, shot: Dict[str, Any]) -> Optional[str]:
        shot_id = shot.get("shot_id", "shot_000")
        matches = sorted(self.shots_dir.glob(f"{shot_id}.*"))
        for m in matches:
            if m.suffix.lower() in (".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv"):
                return str(m)
        return str(matches[0]) if matches else None

    def _extract_last_frame(self, video_path: str, shot_id: str) -> Optional[str]:
        """用 ffmpeg 抽取视频最后一帧作为下一镜头首帧参考图。"""
        frames_dir = self.shots_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out = frames_dir / f"{shot_id}_last_frame.jpg"
        cmd = [
            self.ffmpeg, "-y", "-sseof", "-0.5",
            "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            if out.exists() and out.stat().st_size > 0:
                log.debug("已抽取尾帧：%s", out)
                return str(out)
            log.warning("ffmpeg 抽取尾帧失败：%s", proc.stderr.decode(errors="ignore")[:300])
        except Exception as exc:
            log.warning("ffmpeg 抽取尾帧异常：%s", exc)
        return None
