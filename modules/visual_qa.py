"""生成后画面层视觉质检（v1）。

在镜头生成完成后，抽取相邻同场景镜头的前后帧，计算：
- 正常相似度（漂移检测）
- 水平镜像相似度（镜像检测）
并把结果写入 continuity_report.json 的 visual_qa 段。

本模块是“提示词层状态机”的画面层补充：只做告警，不阻塞、不重生成。
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config_utils import detect_ffmpeg

log = logging.getLogger(__name__)


class VisualQA:
    """轻量画面质检器。"""

    def __init__(self, config: Dict[str, Any], storyboard: Dict[str, Any],
                 shots_dir: str | Path):
        vq = config.get("visual_qa") or {}
        self.enabled = bool(vq.get("enabled", True))
        self.resize = int(vq.get("resize", 64))
        self.hash_size = int(vq.get("hash_size", 16))
        self.drift_threshold = float(vq.get("drift_threshold", 0.85))
        self.mirror_ratio_threshold = float(vq.get("mirror_ratio_threshold", 1.15))
        self.min_file_size_kb = int(vq.get("min_file_size_kb", 10))
        self.ffmpeg = detect_ffmpeg(config)
        self.storyboard = storyboard
        self.shots_dir = Path(shots_dir)
        self.frames_dir = self.shots_dir / "frames" / "qa"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.shot_by_id: Dict[str, Dict[str, Any]] = {
            shot.get("shot_id"): shot
            for shot in (storyboard.get("shots", []) or [])
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """执行质检，返回 visual_qa 段。"""
        transitions: List[Dict[str, Any]] = []
        shots = self.storyboard.get("shots", []) or []
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            if a.get("scene_id") != b.get("scene_id"):
                continue  # 场景切换，不做帧间连续性校验
            entry = self._analyze_transition(a, b)
            if entry is not None:
                transitions.append(entry)

        drift = sum(1 for t in transitions if t.get("drift_warning"))
        mirror = sum(1 for t in transitions if t.get("mirror_warning"))
        summary = {
            "checked_transitions": len(transitions),
            "drift_warnings": drift,
            "mirror_warnings": mirror,
        }
        result = {
            "enabled": self.enabled,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "transitions": transitions,
            "summary": summary,
        }
        log.info("视觉质检完成：检查 %d 组相邻镜头，漂移告警 %d，镜像告警 %d",
                 len(transitions), drift, mirror)
        return result

    # ------------------------------------------------------------------
    # 单组镜头分析
    # ------------------------------------------------------------------
    def _analyze_transition(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        va = self._video_path(a)
        vb = self._video_path(b)
        if not va or not vb:
            return None
        if not self._valid_video(va) or not self._valid_video(vb):
            log.debug("跳过 %s->%s：镜头文件无效", a.get("shot_id"), b.get("shot_id"))
            return None

        fa = self._extract_frame(va, "last")
        fb = self._extract_frame(vb, "first")
        if not fa or not fb:
            return None

        normal_sim = self._image_similarity(fa, fb, flip_b=False)
        flipped_sim = self._image_similarity(fa, fb, flip_b=True)
        mirror_ratio = flipped_sim / max(normal_sim, 1e-6)

        camera_a = str(a.get("camera_side") or "A")
        camera_b = str(b.get("camera_side") or "A")
        reverse = camera_a != camera_b

        drift_warning = False
        mirror_warning = False
        note = ""
        if reverse:
            # 反打镜头：画面本应左右翻转，跳过普通阈值判断
            note = "反打镜头（机位 A/B 切换）：跳过镜像/漂移阈值判断"
        else:
            if normal_sim < self.drift_threshold:
                drift_warning = True
            if mirror_ratio >= self.mirror_ratio_threshold:
                mirror_warning = True
                note = "疑似镜像翻转：翻转相似度明显高于正常相似度"

        return {
            "shot_a": a.get("shot_id"),
            "shot_b": b.get("shot_id"),
            "scene_id": a.get("scene_id"),
            "camera_side_a": camera_a,
            "camera_side_b": camera_b,
            "normal_similarity": round(normal_sim, 4),
            "flipped_similarity": round(flipped_sim, 4),
            "mirror_ratio": round(mirror_ratio, 4),
            "drift_warning": drift_warning,
            "mirror_warning": mirror_warning,
            "note": note,
        }

    # ------------------------------------------------------------------
    # 视频/帧工具
    # ------------------------------------------------------------------
    def _video_path(self, shot: Dict[str, Any]) -> Optional[Path]:
        shot_id = shot.get("shot_id", "")
        matches = sorted(self.shots_dir.glob(f"{shot_id}.*"))
        for m in matches:
            if m.suffix.lower() in (".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv"):
                return m
        return matches[0] if matches else None

    def _valid_video(self, path: Path) -> bool:
        try:
            return path.stat().st_size > self.min_file_size_kb * 1024
        except OSError:
            return False

    def _extract_frame(self, video: Path, kind: str) -> Optional[Path]:
        """抽取首帧或尾帧，缓存到 shots/frames/qa/。"""
        out = self.frames_dir / f"{video.stem}_{kind}.jpg"
        if out.exists() and out.stat().st_size > 0:
            return out
        if kind == "first":
            cmd = [self.ffmpeg, "-y", "-ss", "0.0", "-i", str(video),
                   "-frames:v", "1", "-q:v", "2", str(out)]
        else:
            cmd = [self.ffmpeg, "-y", "-sseof", "-0.5", "-i", str(video),
                   "-frames:v", "1", "-q:v", "2", str(out)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
            if out.exists() and out.stat().st_size > 0:
                return out
            log.warning("抽帧失败：%s %s（%s）", video, kind,
                        proc.stderr.decode(errors="ignore")[:200])
        except Exception as exc:
            log.warning("抽帧异常：%s %s（%s）", video, kind, exc)
        return None

    # ------------------------------------------------------------------
    # 相似度计算
    # ------------------------------------------------------------------
    def _load_pixels(self, path: Path, flip: bool = False) -> List[int]:
        img = Image.open(path).convert("L")
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((self.resize, self.resize), Image.LANCZOS)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return list(img.getdata())

    def _image_similarity(self, path_a: Path, path_b: Path, flip_b: bool = False) -> float:
        try:
            pa = self._load_pixels(path_a)
            pb = self._load_pixels(path_b, flip=flip_b)
        except Exception as exc:
            log.warning("读取帧失败：%s / %s（%s）", path_a, path_b, exc)
            return 0.0
        ahash = 1.0 - self._hamming(self._ahash(pa), self._ahash(pb)) / (self.hash_size ** 2)
        ncc = self._ncc(pa, pb)
        hist = self._hist_corr(path_a, path_b, flip_b)
        # 权重：哈希 0.5 / 像素相关 0.3 / 直方图 0.2
        return max(0.0, min(1.0, 0.5 * ahash + 0.3 * ncc + 0.2 * hist))

    def _ahash(self, pixels: List[int]) -> int:
        n = len(pixels)
        mean = sum(pixels) / max(n, 1)
        bits = 0
        for p in pixels:
            bits = (bits << 1) | (1 if p > mean else 0)
        return bits

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        x = a ^ b
        return bin(x).count("1")

    @staticmethod
    def _ncc(pa: List[int], pb: List[int]) -> float:
        n = len(pa)
        if n == 0:
            return 0.0
        ma = sum(pa) / n
        mb = sum(pb) / n
        num = sum((a - ma) * (b - mb) for a, b in zip(pa, pb))
        da = math.sqrt(sum((a - ma) ** 2 for a in pa))
        db = math.sqrt(sum((b - mb) ** 2 for b in pb))
        if da * db == 0:
            return 0.0
        return num / (da * db)

    def _hist_corr(self, path_a: Path, path_b: Path, flip_b: bool) -> float:
        try:
            ha = Image.open(path_a).convert("L").resize((self.resize, self.resize)).histogram()
            img_b = Image.open(path_b).convert("L").resize((self.resize, self.resize))
            if flip_b:
                img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
            hb = img_b.histogram()
        except Exception:
            return 0.0
        na = sum(ha) or 1
        nb = sum(hb) or 1
        ha = [x / na for x in ha]
        hb = [x / nb for x in hb]
        ma = sum(ha) / len(ha)
        mb = sum(hb) / len(hb)
        num = sum((a - ma) * (b - mb) for a, b in zip(ha, hb))
        da = math.sqrt(sum((a - ma) ** 2 for a in ha))
        db = math.sqrt(sum((b - mb) ** 2 for b in hb))
        if da * db == 0:
            return 0.0
        return num / (da * db)

    # ------------------------------------------------------------------
    # JSON 输出
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.run(), ensure_ascii=False, indent=2)
