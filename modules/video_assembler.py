"""视频拼接模块。

使用 ffmpeg 按 shot_id 顺序拼接镜头：
- 默认 concat demuxer + libx264 重编码，保证分辨率/帧率一致
- 可选 xfade 交叉淡入转场
- 可选烧录旁白字幕（由 storyboard 生成 SRT）
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class VideoAssembler:
    """ffmpeg 视频拼接器。"""

    def __init__(self, config: Dict[str, Any]):
        from .config_utils import detect_ffmpeg
        cfg = config.get("ffmpeg", {}) or {}
        self.ffmpeg = detect_ffmpeg(config)
        self.transitions = bool(cfg.get("transitions", False))
        self.transition_duration = float(cfg.get("transition_duration", 0.5))
        self.subtitles = bool(cfg.get("subtitles", False))
        self.crf = str(cfg.get("crf", 18))
        self.preset = str(cfg.get("preset", "medium"))
        minimax_cfg = config.get("minimax", {}) or {}
        self.fps = str(minimax_cfg.get("fps", 24))
        self._temp_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    def assemble(self, shot_files: List[str], storyboard: Dict[str, Any],
                 output_path: str | Path) -> Path:
        """拼接所有镜头为 final.mp4。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not shot_files:
            raise ValueError("没有可拼接的镜头文件。")
        if len(shot_files) == 1:
            log.info("仅一个镜头，直接转封装为 %s", output_path)
            self._run([self.ffmpeg, "-y", "-i", shot_files[0],
                       "-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
                       "-pix_fmt", "yuv420p", "-r", self.fps, "-an", str(output_path)])
            return output_path

        srt_path = None
        if self.subtitles:
            srt_path = output_path.with_suffix(".srt")
            self._write_srt(storyboard, srt_path)

        if self.transitions:
            durations = self._collect_durations(shot_files, storyboard)
            if all(d > self.transition_duration + 0.2 for d in durations):
                log.info("使用 xfade 转场拼接 %d 个镜头", len(shot_files))
                self._assemble_with_xfade(shot_files, durations, output_path, srt_path)
                return output_path
            log.warning("存在时长过短的镜头，转场回退为普通拼接。")

        log.info("使用 concat 拼接 %d 个镜头", len(shot_files))
        try:
            self._assemble_concat(shot_files, output_path, srt_path)
        except subprocess.CalledProcessError:
            log.warning("concat 直接重编码失败，回退为“先归一化再拼接”。")
            self._assemble_normalized(shot_files, output_path, srt_path)
        return output_path

    # ------------------------------------------------------------------
    def _assemble_concat(self, files: List[str], out: Path, srt_path: Optional[Path]) -> None:
        list_file = out.with_suffix(".concat.txt")
        list_file.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files),
                             encoding="utf-8")
        vf = self._build_vf(srt_path)
        cmd = [self.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
                "-pix_fmt", "yuv420p", "-r", self.fps, "-c:a", "aac",
                "-b:a", "192k", str(out)]
        self._run(cmd)
        list_file.unlink(missing_ok=True)

    def _assemble_normalized(self, files: List[str], out: Path,
                             srt_path: Optional[Path]) -> None:
        self._temp_dir = out.parent / "temp_normalize"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        normalized: List[str] = []
        for i, f in enumerate(files, 1):
            norm = self._temp_dir / f"norm_{i:03d}.mp4"
            self._run([self.ffmpeg, "-y", "-i", f,
                       "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                       "-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
                       "-pix_fmt", "yuv420p", "-r", self.fps, "-an", str(norm)])
            normalized.append(str(norm))
        list_file = self._temp_dir / "concat.txt"
        list_file.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in normalized),
                             encoding="utf-8")
        cmd = [self.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
        if srt_path:
            cmd += ["-vf", self._escape_subtitles_filter(srt_path)]
        cmd += ["-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)]
        self._run(cmd)
        list_file.unlink(missing_ok=True)

    def _assemble_with_xfade(self, files: List[str], durations: List[float],
                             out: Path, srt_path: Optional[Path]) -> None:
        fd = self.transition_duration
        inputs: List[str] = []
        for f in files:
            inputs += ["-i", f]

        parts: List[str] = []
        for i in range(len(files)):
            parts.append(
                f"[{i}:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={self.fps},"
                f"setpts=PTS-STARTPTS[v{i}]"
            )
        prev = "v0"
        offset = max(0.1, durations[0] - fd)
        for i in range(1, len(files)):
            next_label = f"x{i}"
            if i == len(files) - 1:
                next_label = "vout"
            parts.append(
                f"[{prev}][v{i}]xfade=transition=fade:duration={fd}:offset={offset:.3f}[{next_label}]"
            )
            prev = next_label
            offset = offset + durations[i] - fd

        filter_complex = ";".join(parts)
        if srt_path:
            filter_complex += f";[vout]subtitles={self._escape_subtitles_path(srt_path)}[voutsub]"
            map_label = "[voutsub]"
        else:
            map_label = "[vout]"

        cmd = [self.ffmpeg, "-y", *inputs,
               "-filter_complex", filter_complex,
               "-map", map_label,
               "-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
               "-pix_fmt", "yuv420p", "-an", str(out)]
        self._run(cmd)

    # ------------------------------------------------------------------
    def _build_vf(self, srt_path: Optional[Path]) -> Optional[str]:
        filters = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        if srt_path:
            filters.append(f"subtitles={self._escape_subtitles_path(srt_path)}")
        return ",".join(filters)

    def _escape_subtitles_filter(self, srt_path: Path) -> str:
        return f"subtitles={self._escape_subtitles_path(srt_path)}"

    @staticmethod
    def _escape_subtitles_path(path: Path) -> str:
        # ffmpeg subtitles 过滤器路径转义（Windows）
        s = str(path.resolve()).replace("\\", "/")
        s = s.replace(":", "\\:")
        return s

    # ------------------------------------------------------------------
    def _write_srt(self, storyboard: Dict[str, Any], srt_path: Path) -> None:
        lines: List[str] = []
        t = 0.0
        for i, shot in enumerate(storyboard.get("shots", []) or [], 1):
            narration = str(shot.get("narration") or shot.get("action") or "").strip()
            if not narration:
                continue
            duration = float(shot.get("duration_seconds", 5) or 5)
            start = t
            end = t + duration
            lines.append(str(i))
            lines.append(f"{self._fmt_time(start)} --> {self._fmt_time(end)}")
            lines.append(narration)
            lines.append("")
            t = end
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("已生成字幕：%s", srt_path)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _collect_durations(self, files: List[str], storyboard: Dict[str, Any]) -> List[float]:
        shot_duration = {
            shot.get("shot_id"): float(shot.get("duration_seconds", 5) or 5)
            for shot in storyboard.get("shots", []) or []
        }
        durations: List[float] = []
        for f in files:
            shot_id = Path(f).stem.split("_last_frame")[0]
            durations.append(shot_duration.get(shot_id, 5.0))
        return durations

    # ------------------------------------------------------------------
    def _run(self, cmd: List[str]) -> None:
        log.debug("执行：%s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"找不到 ffmpeg 可执行文件：{self.ffmpeg}。请安装 ffmpeg 并加入 PATH，"
                f"或在 config.yaml 的 ffmpeg.path 中指定完整路径。") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="ignore")[-1500:]
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr)
