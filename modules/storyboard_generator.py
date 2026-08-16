"""自动分镜模块。

调用 OpenAI 兼容 LLM 将小说文本切分为场景与镜头，生成完整分镜表。
支持两种分段策略：
- auto: LLM 根据剧情智能切分，单镜头时长受 max_shot_duration 限制
- fixed_duration: 按用户指定时长（如 10 秒）切分，长句/长场景自动拆分，
  所有镜头总时长覆盖全文
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .continuity_tracker import ContinuityTracker

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位资深电影分镜师、视频生成提示词专家和空间连续性监督员。
你的任务是把小说文本转成可直接用于 MiniMax H3 视频生成的电影分镜表。

语言要求：除 narration 必须逐字引用原文外，其余所有字段（location、characters、
action、camera、start_frame_prompt、end_frame_prompt、spatial_anchors 等）一律用
英文输出；video_prompt 中画面描述、镜头语言、空间锚点全部为英文。

必须遵循电影语言规则：
1. 景别要有变化（远景/全景/中景/近景/特写），机位交代清楚。
2. 运动方向、视线关系、180 度规则必须自洽。
3. 同一场景内，固定设施（门、窗、楼梯、主要家具）的屏幕方位在连续镜头中必须一致。
4. 角色运动方向在连续动作中必须一致：上一镜头从左向右走，下一镜头不能突然从右向左，
   除非剧情需要（如掉头）并在 action 中显式说明。
5. 反打镜头必须显式说明并遵循 180 度规则，例如 camera 写“反打镜头：摄影机移到对面”。
6. 禁止用镜像画面衔接。

每个镜头必须包含以下字段：
{
  "shot_id": "shot_001",
  "scene_id": "scene_01",
  "location": "堂屋",
  "characters": ["林晚"],
  "action": "镜头动作描述（含景别与画面内容）",
  "camera": "机位描述（如：中景，平视，机位A侧，门在画面左侧）",
  "narration": "本镜头覆盖的原文旁白/对白，必须逐字取自原文，按顺序完整覆盖全文、不遗漏、不重复",
  "duration_seconds": 8,
  "start_frame_prompt": "首帧画面描述（含空间方位）",
  "end_frame_prompt": "尾帧画面描述（含空间方位）",
  "spatial_anchors": {"door": "left", "window": "right", "table": "center",
                       "character_facing": "right", "motion_direction": "left_to_right"},
  "scene_layout": {"fixtures": {"door": "left", "window": "right", "table": "center"}},
  "continuity_input": {},
  "continuity_output": {"characters": {"林晚": {"position": "center", "facing": "right",
                                                 "motion_direction": "static"}}},
  "reference_image": "",
  "video_prompt": ""
}

只输出 JSON，不要输出任何解释文字。"""


class StoryboardGenerator:
    """分镜生成器。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_cfg = config.get("llm", {})
        self.storyboard_cfg = config.get("storyboard", {})
        self.minimax_cfg = config.get("minimax", {})
        self._openai_client = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def generate(self, text: str, mode: Optional[str] = None,
                 segment_duration: Optional[float] = None) -> Dict[str, Any]:
        """生成完整分镜表。"""
        text = self._clean_text(text)
        if not text:
            raise ValueError("小说文本为空，无法生成分镜。")

        mode = mode or self.storyboard_cfg.get("mode", "auto")
        segment_duration = float(segment_duration or self.storyboard_cfg.get(
            "segment_duration_seconds", self.minimax_cfg.get("max_shot_duration", 10)))
        max_shot_duration = float(self.minimax_cfg.get("max_shot_duration", 10))

        if mode in ("fixed_duration", "fixed"):
            shots = self._fixed_shots(text, segment_duration)
            effective_mode = "fixed_duration"
            duration_cap = segment_duration
        else:
            shots = self._auto_shots(text, max_shot_duration)
            effective_mode = "auto"
            duration_cap = max_shot_duration

        shots = self._normalize_shots(shots, duration_cap=duration_cap)
        shots = self._apply_continuity(shots)

        storyboard = {
            "meta": {
                "mode": effective_mode,
                "segment_duration_seconds": segment_duration,
                "max_shot_duration": max_shot_duration,
                "total_shots": len(shots),
                "total_duration_seconds": round(sum(float(s.get("duration_seconds", 0)) for s in shots), 2),
                "source_length_chars": len(text),
            },
            "shots": shots,
        }
        log.info("分镜生成完成：mode=%s, shots=%d, 总时长=%.1fs",
                 effective_mode, len(shots), storyboard["meta"]["total_duration_seconds"])
        return storyboard

    # ------------------------------------------------------------------
    # 两种分段策略
    # ------------------------------------------------------------------
    def _auto_shots(self, text: str, max_shot_duration: float) -> List[Dict[str, Any]]:
        """auto 模式：LLM 智能切分。"""
        target = self.storyboard_cfg.get("target_duration_seconds")
        target_line = ""
        if target:
            target_line = (
                f"0. 全片总时长控制在 {float(target):.0f} 秒左右，"
                f"镜头数尽量精简（每镜 4-10 秒）。\n"
            )
        user = (
            f"请将下面这部小说切分为连续镜头。要求：\n"
            f"{target_line}"
            f"1. 每个镜头 duration_seconds <= {max_shot_duration} 秒。\n"
            f"2. 所有镜头的 narration 字段必须按顺序、逐字、完整覆盖原文（可跨镜头断句，但不得增删改原文）。\n"
            f"3. 同一场景的 scene_layout 必须一致，spatial_anchors 中的固定设施屏幕方位必须与 scene_layout 一致。\n"
            f"4. 相邻镜头之间，continuity_output 必须是 continuity_input 的合理演化；"
            f"首个镜头 continuity_input 留空 {{}}。\n\n"
            f"小说原文：\n{text}"
        )
        data = self._call_llm(SYSTEM_PROMPT, user)
        shots = (data or {}).get("shots") or []
        if shots:
            log.info("auto 模式：LLM 切分出 %d 个镜头", len(shots))
            return shots
        log.warning("auto 模式：LLM 未返回有效分镜，回退为固定时长切分。")
        return self._fixed_shots(text, max_shot_duration)

    def _fixed_shots(self, text: str, segment_duration: float) -> List[Dict[str, Any]]:
        """fixed_duration 模式：按指定时长切分文本为若干镜头。"""
        chunks = self._split_text_by_duration(text, segment_duration)
        log.info("fixed_duration 模式：%.1fs/镜头，切分为 %d 个文本段", segment_duration, len(chunks))

        user = (
            f"下面有 {len(chunks)} 段文本，每段是一个镜头的 narration（必须逐字保留该段文本）。\n"
            f"请为每段文本生成一个镜头，所有镜头 narration 必须按编号顺序与给定文本完全一致。\n"
            f"duration_seconds = 该段文本字数 / 5 秒，不足 2 秒按 2 秒，最长不超过 {segment_duration} 秒。\n\n"
        )
        for i, chunk in enumerate(chunks, 1):
            user += f"--- 第 {i} 段 ---\n{chunk}\n"
        user += "\n请输出 JSON：{\"shots\": [...]}"

        data = self._call_llm(SYSTEM_PROMPT, user)
        shots = (data or {}).get("shots") or []
        if len(shots) == len(chunks):
            log.info("fixed_duration 模式：LLM 为 %d 个文本段生成了镜头", len(shots))
            return shots
        log.warning("fixed_duration 模式：LLM 返回数量不匹配（%d/%d），使用本地确定性分镜。",
                    len(shots), len(chunks))
        return self._fallback_shots(chunks, segment_duration)

    # ------------------------------------------------------------------
    # 本地回退分镜
    # ------------------------------------------------------------------
    def _fallback_shots(self, chunks: List[str], segment_duration: float) -> List[Dict[str, Any]]:
        cps = float(self.storyboard_cfg.get("chars_per_second", 5))
        shots: List[Dict[str, Any]] = []
        scene_counter = 0
        location_map: Dict[str, str] = {}
        prev_location: Optional[str] = None
        prev_characters: List[str] = []

        for idx, chunk in enumerate(chunks, 1):
            location = self._guess_location(chunk, idx)
            if location.startswith("地点") and prev_location:
                location = prev_location  # 未识别到新地点时继承上一镜头场景
            else:
                prev_location = location
            if location not in location_map:
                scene_counter += 1
                location_map[location] = f"scene_{scene_counter:02d}"
            scene_id = location_map[location]
            characters = self._guess_characters(chunk)
            if characters == ["主角"] and prev_characters:
                characters = prev_characters
            else:
                prev_characters = characters
            sentences = [s.strip() for s in re.split(r'(?<=[。！？!？；;…])', chunk) if s.strip()]
            first_sentence = sentences[0] if sentences else chunk[:50]
            last_sentence = sentences[-1] if sentences else chunk[-50:]
            duration = max(2.0, min(float(segment_duration), round(len(chunk) / cps, 1)))

            shot = {
                "shot_id": f"shot_{idx:03d}",
                "scene_id": scene_id,
                "location": location,
                "characters": characters,
                "action": first_sentence,
                "camera": "中景，平视，机位A侧（门在画面左侧，窗在画面右侧）",
                "narration": chunk,
                "duration_seconds": duration,
                "start_frame_prompt": f"{first_sentence} 门在画面左侧，窗户在画面右侧，桌子在画面中央。",
                "end_frame_prompt": f"{last_sentence} 门在画面左侧，窗户在画面右侧，桌子在画面中央。",
                "spatial_anchors": {
                    "door": "left",
                    "window": "right",
                    "table": "center",
                    "character_facing": "right",
                    "motion_direction": "left_to_right",
                },
                "scene_layout": {"fixtures": {"door": "left", "window": "right", "table": "center"}},
                "continuity_input": {},
                "continuity_output": {},
                "reference_image": "",
                "video_prompt": "",
            }
            shots.append(shot)
        return shots

    # ------------------------------------------------------------------
    # 归一化与连续性应用
    # ------------------------------------------------------------------
    def _normalize_shots(self, shots: List[Dict[str, Any]],
                         duration_cap: Optional[float] = None) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if duration_cap is None:
            duration_cap = float(self.minimax_cfg.get("max_shot_duration", 10))
        location_map: Dict[str, str] = {}
        scene_counter = 0

        for idx, raw in enumerate(shots, 1):
            shot = dict(raw or {})
            shot["shot_id"] = shot.get("shot_id") or f"shot_{idx:03d}"
            location = shot.get("location") or "未命名地点"
            if location not in location_map:
                scene_counter += 1
                location_map[location] = f"scene_{scene_counter:02d}"
            shot["scene_id"] = shot.get("scene_id") or location_map[location]

            chars = shot.get("characters") or []
            if isinstance(chars, str):
                chars = re.split(r'[、,，\s]+', chars.strip()) if chars.strip() else []
            shot["characters"] = [c for c in chars if c]

            try:
                duration = float(shot.get("duration_seconds", 0) or 0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                narration = shot.get("narration") or ""
                cps = float(self.storyboard_cfg.get("chars_per_second", 5))
                duration = max(2.0, round(len(narration) / cps, 1))
            shot["duration_seconds"] = min(duration, duration_cap) if duration_cap > 0 else duration

            shot.setdefault("spatial_anchors", {})
            if not isinstance(shot.get("spatial_anchors"), dict):
                shot["spatial_anchors"] = {}
            shot.setdefault("scene_layout", {"fixtures": {}})
            if not isinstance(shot.get("scene_layout"), dict):
                shot["scene_layout"] = {"fixtures": {}}
            shot["scene_layout"].setdefault("fixtures", {})

            shot.setdefault("continuity_input", {})
            shot.setdefault("continuity_output", {})
            shot.setdefault("reference_image", "")
            shot.setdefault("video_prompt", "")
            shot.setdefault("action", shot.get("narration", "")[:80])
            shot.setdefault("camera", "中景，平视，机位A侧")
            shot.setdefault("start_frame_prompt", shot.get("action", ""))
            shot.setdefault("end_frame_prompt", shot.get("action", ""))
            shot.setdefault("dialogue", "")
            shot.setdefault("narration", "")
            normalized.append(shot)
        return normalized

    def _apply_continuity(self, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """顺序跑一遍 ContinuityTracker：填充/修正 spatial_anchors、continuity_input/output、video_prompt。"""
        tracker = ContinuityTracker()
        for shot in shots:
            tracker.prepare_shot(shot)
            tracker.apply_shot(shot)
        self.last_report = tracker.report
        self.last_report["summary"] = {
            "shots_total": len(shots),
            "conflicts": len(tracker.report.get("conflicts", [])),
            "corrections": len(tracker.report.get("corrections", [])),
            "reverse_shots": len(tracker.report.get("reverse_shots", [])),
        }
        return shots

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    def _call_llm(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        if not self.llm_cfg.get("api_key"):
            log.info("未配置 llm.api_key，跳过 LLM 调用，使用本地确定性分镜。")
            return None
        try:
            from openai import OpenAI  # 延迟导入，保证无 openai 时仍可本地分镜
        except Exception as exc:  # pragma: no cover
            log.warning("openai 库不可用（%s），使用本地确定性分镜。", exc)
            return None

        try:
            client = OpenAI(
                api_key=self.llm_cfg.get("api_key", ""),
                base_url=self.llm_cfg.get("base_url", "https://api.openai.com/v1"),
                timeout=float(self.llm_cfg.get("timeout_seconds", 120)),
            )
            kwargs: Dict[str, Any] = {
                "model": self.llm_cfg.get("model", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": float(self.llm_cfg.get("temperature", 0.4)),
                "max_tokens": int(self.llm_cfg.get("max_tokens", 64000)),
            }
            # 可选随机种子，便于复现分镜
            seed = self.llm_cfg.get("seed")
            if seed not in (None, "", "null"):
                try:
                    kwargs["seed"] = int(seed)
                except (TypeError, ValueError):
                    log.warning("llm.seed 不是整数，已忽略：%s", seed)
            try:
                kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)

            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                log.warning("LLM 输出达到 max_tokens 上限被截断，分镜可能不完整，建议增大 llm.max_tokens。")
            content = (choice.message.content or "").strip()
            if not content:
                log.warning("LLM 返回空 content（推理模型可能把预算花在 reasoning 上），回退本地分镜。")
                return None
            return self._parse_json(content)
        except Exception as exc:
            log.warning("LLM 调用失败：%s", exc)
            return None

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
        content = content.strip()
        # 容忍前后附加说明文字 / 多段 JSON：只提取第一个完整 JSON 对象
        candidate = StoryboardGenerator._extract_first_json_object(content)
        if candidate is None:
            candidate = content
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            log.warning("LLM 返回的 JSON 解析失败：%s\n原文前 500 字：%s",
                        exc, content[:500])
            return None

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """提取文本中第一个完整 JSON 对象字符串。

        正确处理字符串内的花括号与转义；遇到第一个配平的 '}' 即返回，
        因此 LLM 输出多个对象/前后缀时不会触发 Extra data。
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    # ------------------------------------------------------------------
    # 文本切分
    # ------------------------------------------------------------------
    def _split_text_by_duration(self, text: str, duration: float) -> List[str]:
        cps = float(self.storyboard_cfg.get("chars_per_second", 5))
        target = max(10, int(duration * cps))
        sentences = [s.strip() for s in re.split(r'(?<=[。！？!？；;…])', text) if s.strip()]
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if len(sent) > target:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_sentence(sent, target))
            else:
                if len(current) + len(sent) <= target:
                    current += sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_long_sentence(sentence: str, target: int) -> List[str]:
        clauses = [c for c in re.split(r'(?<=[，,：:])', sentence) if c]
        pieces: List[str] = []
        current = ""
        for clause in clauses:
            if len(clause) > target:
                if current:
                    pieces.append(current)
                    current = ""
                for i in range(0, len(clause), target):
                    pieces.append(clause[i:i + target])
            else:
                if len(current) + len(clause) <= target:
                    current += clause
                else:
                    if current:
                        pieces.append(current)
                    current = clause
        if current:
            pieces.append(current)
        return [p for p in pieces if p]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\ufeff", "")  # 去掉 UTF-8 BOM
        text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)  # 去掉 markdown 标题标记
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)      # 去掉代码块
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _guess_location(chunk: str, idx: int) -> str:
        for key in ("堂屋", "院子", "书房", "卧室", "厨房", "街道", "客栈", "大殿", "走廊", "楼梯"):
            if key in chunk:
                return key
        return f"地点{idx}"

    @staticmethod
    def _guess_characters(chunk: str) -> List[str]:
        # 简单启发式：取常见中文名（2-3 字，前后有动词/标点），此处回退为常见人物名
        names = ["林晚"]
        found = [n for n in names if n in chunk]
        return found or ["主角"]

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save(self, storyboard: Dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, ensure_ascii=False, indent=2)
        log.info("分镜表已保存：%s", path)
