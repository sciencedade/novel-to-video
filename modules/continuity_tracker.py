"""场景空间连续性状态机（核心模块）。

维护场景级固定设施布局与角色级空间状态，在镜头间传递并校验：
- 固定设施（门/窗/家具）屏幕方位一致性
- 角色朝向与运动方向连续性（方向锁）
- 反打镜头与 180 度规则
- 镜像画面检测与修正
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

LEFT_RIGHT_KEYS = ("left", "right")


def flip_horizontal(value: str) -> str:
    """翻转屏幕水平方位。"""
    if not isinstance(value, str):
        return value
    v = value.lower().strip()
    table = {
        "left": "right",
        "right": "left",
        "left_front": "right_front",
        "right_front": "left_front",
        "left_back": "right_back",
        "right_back": "left_back",
        "front_left": "front_right",
        "front_right": "front_left",
        "back_left": "back_right",
        "back_right": "back_left",
    }
    return table.get(v, v)


def flip_motion_direction(value: str) -> str:
    """翻转屏幕运动方向（机位切换到对面时使用）。"""
    if not isinstance(value, str):
        return value
    v = value.lower().strip()
    table = {
        "left_to_right": "right_to_left",
        "right_to_left": "left_to_right",
        "toward_camera": "away_from_camera",
        "away_from_camera": "toward_camera",
    }
    return table.get(v, v)


def _mirror_of_anchors(anchors: Dict[str, Any]) -> Dict[str, Any]:
    """返回将锚点中所有水平方位镜像后的副本（用于镜像检测）。"""
    mirrored: Dict[str, Any] = {}
    for key, value in anchors.items():
        if isinstance(value, str) and value.lower().strip() in (
            "left", "right", "left_front", "right_front",
            "left_back", "right_back", "front_left", "front_right",
            "back_left", "back_right",
        ):
            mirrored[key] = flip_horizontal(value)
        else:
            mirrored[key] = value
    return mirrored


class ContinuityTracker:
    """场景空间连续性跟踪器。

    维护三类状态：
    1. scene_layouts: 场景级固定设施世界坐标（门/窗/家具 左/中/右/前/后）
    2. character_states: 角色位置、朝向、运动方向
    3. camera_state: 每个场景当前机位侧（A=主侧，B=对面/反打）
    """

    def __init__(self) -> None:
        self.scene_layouts: Dict[str, Dict[str, Any]] = {}
        self.character_states: Dict[str, Dict[str, Any]] = {}
        self.camera_state: Dict[str, str] = {}
        self.shot_history: List[Dict[str, Any]] = []
        self.report: Dict[str, Any] = {
            "conflicts": [],
            "corrections": [],
            "reverse_shots": [],
            "summary": {},
        }

    # ------------------------------------------------------------------
    # 初始化与场景布局
    # ------------------------------------------------------------------
    def initialize_from_storyboard(self, storyboard: Dict[str, Any]) -> None:
        """从分镜表初始化场景布局与初始角色状态（用于断点续传或校验）。"""
        shots = storyboard.get("shots", []) or []
        for shot in shots:
            self._ensure_layout(shot)
        if not shots:
            return
        first = shots[0]
        self.camera_state.setdefault(first.get("scene_id", "scene_01"), "A")
        inherited = (first.get("continuity_input") or {}).get("characters", {})
        for char in first.get("characters", []) or []:
            if char in inherited:
                self.character_states[char] = dict(inherited[char])
            else:
                self.character_states[char] = self._infer_character_state(first, char)

    def _ensure_layout(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """确保场景布局存在；若镜头携带新的固定设施则合并进场景布局。"""
        scene_id = shot.get("scene_id") or "scene_01"
        shot["scene_id"] = scene_id

        layout = shot.get("scene_layout") or {}
        fixtures = dict(layout.get("fixtures") or {})

        if not fixtures:
            # 从 spatial_anchors 中提取固定设施（排除角色类锚点）
            anchors = shot.get("spatial_anchors") or {}
            excluded = {"character_facing", "motion_direction", "character_position",
                        "camera_side", "character_screen_direction"}
            for name, pos in anchors.items():
                if name not in excluded and isinstance(pos, str):
                    fixtures[name] = pos
        if not fixtures:
            # 默认堂屋/房间布局
            fixtures = {"door": "left", "window": "right", "table": "center"}

        if scene_id not in self.scene_layouts:
            self.scene_layouts[scene_id] = {
                "fixtures": fixtures,
                "world_axis": layout.get("world_axis", "camera_A"),
            }
        else:
            for name, pos in fixtures.items():
                self.scene_layouts[scene_id]["fixtures"].setdefault(name, pos)

        shot["scene_layout"] = {
            "scene_id": scene_id,
            "fixtures": copy.deepcopy(self.scene_layouts[scene_id]["fixtures"]),
            "world_axis": self.scene_layouts[scene_id]["world_axis"],
        }
        return self.scene_layouts[scene_id]

    # ------------------------------------------------------------------
    # 镜头准备（生成前校验/修正）
    # ------------------------------------------------------------------
    def prepare_shot(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """镜头生成前调用：读取当前状态 -> 校验/修正 spatial_anchors -> 注入提示词。

        注意：该方法会就地修改传入的 shot 字典（填充/修正 spatial_anchors、
        continuity_input、video_prompt 等），并返回该字典。
        """
        if shot is None:
            shot = {}
        scene_id = shot.setdefault("scene_id", "scene_01")
        layout = self._ensure_layout(shot)

        prev_camera = self.camera_state.get(scene_id, "A")
        camera_side = self._detect_camera_side(shot, default=prev_camera)
        reverse = camera_side != prev_camera
        explicit_reverse = self._is_explicit_reverse(shot)

        # 方向锁：未显式说明反打时，不允许机位翻面
        if reverse and not explicit_reverse:
            self._record(
                "corrections", shot,
                "camera_flip_without_annotation",
                f"场景 {scene_id} 机位从 {prev_camera} 侧翻转到 {camera_side} 侧但未标注反打镜头，"
                f"已锁回 {prev_camera} 侧以保持屏幕方向一致。",
                corrected=True,
            )
            camera_side = prev_camera
            shot["camera_side"] = camera_side
            reverse = False
        elif reverse and explicit_reverse:
            self._record(
                "reverse_shots", shot,
                "reverse_shot_annotated",
                f"场景 {scene_id} 显式反打镜头：机位从 {prev_camera} 侧移到 {camera_side} 侧，"
                f"屏幕左右方位将按 180 度规则翻转并在提示词中显式标注。",
                corrected=False,
            )
        shot["camera_side"] = camera_side

        # 1) 固定设施锚点校验：与场景布局的期望屏幕方位比对
        expected_positions = self._expected_screen_positions(layout, camera_side)
        anchors = shot.setdefault("spatial_anchors", {})
        raw_anchors = copy.deepcopy(anchors)

        for fixture_name, expected_pos in expected_positions.items():
            if fixture_name not in anchors:
                anchors[fixture_name] = expected_pos
                self._record(
                    "corrections", shot,
                    "anchor_added",
                    f"为镜头补充固定设施锚点 {fixture_name}={expected_pos}。",
                    corrected=True,
                )
            elif anchors[fixture_name] != expected_pos:
                if reverse and explicit_reverse:
                    self._record(
                        "reverse_shots", shot,
                        "anchor_flipped_by_reverse_shot",
                        f"反打镜头：固定设施 {fixture_name} 屏幕方位由 "
                        f"{flip_horizontal(expected_pos)} 翻转为 {expected_pos}，符合机位说明。",
                        corrected=False,
                    )
                else:
                    self._record(
                        "conflicts", shot,
                        "fixture_anchor_conflict",
                        f"固定设施 {fixture_name} 锚点为 {anchors[fixture_name]}，"
                        f"与场景布局期望的 {expected_pos} 冲突，已自动修正。",
                        corrected=True,
                    )
                    anchors[fixture_name] = expected_pos

        # 2) 镜像检测：同场景、未反打时，锚点整体镜像会被视为非法镜像画面
        if self.shot_history and not reverse:
            prev_shot = self.shot_history[-1]
            prev_anchors = prev_shot.get("spatial_anchors") or {}
            if prev_shot.get("scene_id") == scene_id and prev_anchors:
                mirrored = _mirror_of_anchors(raw_anchors)
                common = [k for k in prev_anchors if k in mirrored]
                if common and all(prev_anchors[k] == mirrored[k] for k in common) \
                        and any(prev_anchors[k] != raw_anchors.get(k) for k in common):
                    self._record(
                        "conflicts", shot,
                        "possible_mirror",
                        f"镜头锚点与上一镜头呈镜像翻转但未标注反打镜头，已按场景布局修正，"
                        f"禁止镜像画面衔接。",
                        corrected=True,
                    )

        # 3) 角色运动方向锁
        for char in shot.get("characters", []) or []:
            self._validate_character_motion(shot, char, scene_id, camera_side, reverse, explicit_reverse)

        # 4) 生成 continuity_input（继承状态快照）
        shot["continuity_input"] = self._build_continuity_input(shot, scene_id, prev_camera)

        # 5) 注入完整提示词
        shot["video_prompt"] = self.build_video_prompt(shot)

        # 同步场景布局到镜头（保证 storyboard.json 中可见）
        shot["scene_layout"] = {
            "scene_id": scene_id,
            "fixtures": copy.deepcopy(self.scene_layouts[scene_id]["fixtures"]),
            "world_axis": self.scene_layouts[scene_id]["world_axis"],
        }
        return shot

    # ------------------------------------------------------------------
    # 镜头结束（更新状态）
    # ------------------------------------------------------------------
    def apply_shot(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """镜头生成完成后调用：把镜头结束状态写入跟踪器。"""
        scene_id = shot.get("scene_id", "scene_01")
        camera_side = shot.get("camera_side") or self._detect_camera_side(shot, self.camera_state.get(scene_id, "A"))
        self.camera_state[scene_id] = camera_side

        out = shot.get("continuity_output") or {}
        out_chars = out.get("characters", {}) or {}
        final_chars: Dict[str, Dict[str, Any]] = {}
        for char in shot.get("characters", []) or []:
            char_out = out_chars.get(char, {})
            state = {
                "scene_id": scene_id,
                "position": char_out.get("position")
                or shot["spatial_anchors"].get("character_position")
                or shot["spatial_anchors"].get(f"{char}_position")
                or self.character_states.get(char, {}).get("position", "center"),
                "facing": char_out.get("facing")
                or shot["spatial_anchors"].get("character_facing")
                or shot["spatial_anchors"].get(f"{char}_facing")
                or self.character_states.get(char, {}).get("facing", "right"),
                "motion_direction": char_out.get("motion_direction")
                or shot["spatial_anchors"].get("motion_direction")
                or shot["spatial_anchors"].get(f"{char}_motion_direction")
                or self.character_states.get(char, {}).get("motion_direction", "static"),
            }
            # 角色脱离该场景时（剧情离场），保留离场标记
            if char_out.get("exits") or char_out.get("offscreen"):
                state["exits"] = True
            self.character_states[char] = state
            final_chars[char] = copy.deepcopy(state)

        shot["continuity_output"] = {
            "scene_id": scene_id,
            "camera_side": camera_side,
            "characters": final_chars,
            "fixtures": copy.deepcopy(self.scene_layouts[scene_id]["fixtures"]),
        }
        self.shot_history.append(shot)
        return shot

    # ------------------------------------------------------------------
    # 提示词构建
    # ------------------------------------------------------------------
    def build_video_prompt(self, shot: Dict[str, Any], base_prompt: Optional[str] = None) -> str:
        """构建注入 MiniMax H3 的完整提示词，显式锁定空间关系。"""
        parts: List[str] = []
        style = shot.get("style") or ""
        if base_prompt:
            parts.append(base_prompt)
        else:
            if shot.get("narration"):
                parts.append(f"旁白/对白：{shot['narration']}")
            if shot.get("action"):
                parts.append(f"画面内容：{shot['action']}")
            if shot.get("camera"):
                parts.append(f"镜头语言：{shot['camera']}")
            if shot.get("start_frame_prompt"):
                parts.append(f"首帧画面：{shot['start_frame_prompt']}")
            if shot.get("end_frame_prompt"):
                parts.append(f"尾帧画面：{shot['end_frame_prompt']}")
            if style:
                parts.append(f"风格：{style}")

        # 空间连续性锁定
        lock: List[str] = ["【空间连续性锁定】"]
        anchors = shot.get("spatial_anchors") or {}
        layout = (shot.get("scene_layout") or {}).get("fixtures", {})
        if layout:
            fixture_text = "，".join(f"{name}在画面{self._cn(pos)}" for name, pos in layout.items())
            lock.append(f"场景固定设施屏幕方位：{fixture_text}。")

        inherited = shot.get("continuity_input") or {}
        if inherited.get("inherited_from"):
            lock.append(
                f"延续上一镜头（{inherited['inherited_from']}）：机位为 {inherited.get('camera_side', 'A')} 侧，"
                f"固定设施屏幕方位保持不变。"
            )

        for char in shot.get("characters", []) or []:
            facing = anchors.get("character_facing") or anchors.get(f"{char}_facing") or "right"
            motion = anchors.get("motion_direction") or "static"
            pos = anchors.get("character_position") or anchors.get(f"{char}_position") or "center"
            lock.append(
                f"角色 {char}：位于画面{self._cn(pos)}，面向{self._cn(facing)}，"
                f"运动方向{self._cn(motion)}。"
            )

        # 角色定妆描述（提示词锁定第 2 层）
        char_refs = shot.get("character_references") or {}
        for char, info in char_refs.items():
            desc = (info or {}).get("description") or ""
            if desc:
                lock.append(f"角色 {char} 定妆描述：{desc}。")

        # 禁止变化项（负面提示词锁定）
        forbidden = shot.get("forbidden_changes") or []
        if forbidden:
            lock.append("禁止变化项：" + "，".join(forbidden) + "。")

        camera_side = shot.get("camera_side") or "A"
        if camera_side == "B":
            lock.append(
                "反打镜头：摄影机已移到对面，画面左右方位相对上一镜头翻转"
                "（符合 180 度规则说明），固定设施与角色屏幕位置以上述锁定为准。"
            )

        lock.append("禁止镜像画面。禁止左右方位跳变。空间关系以上述锁定为准。")
        parts.extend(lock)

        if not base_prompt:
            parts.append(f"画面风格：{style or 'cinematic, coherent, high quality'}")
        return "\n".join(p for p in parts if p)

    def retry_adjustment(self, shot: Dict[str, Any], attempt: int, error: str) -> str:
        """生成失败重试时附加的提示词修正文本。"""
        anchors = shot.get("spatial_anchors") or {}
        camera_side = shot.get("camera_side", "A")
        lines = [
            f"【重试修正 第 {attempt} 次】上一次生成失败：{str(error)[:200]}。",
            "请严格保持以下空间锚点，不得镜像画面，不得跳变：",
            json.dumps(anchors, ensure_ascii=False),
        ]
        if camera_side == "B":
            lines.append("本镜头为反打镜头：摄影机在对面，画面左右方位已按 180 度规则翻转。")
        else:
            lines.append("本镜头与上一镜头机位相同：固定设施屏幕方位和角色运动方向必须延续，不得翻转。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 校验与报告
    # ------------------------------------------------------------------
    def validate_storyboard(self, storyboard: Dict[str, Any]) -> Dict[str, Any]:
        """对完整分镜表做一次顺序模拟校验，返回报告。"""
        fresh = ContinuityTracker()
        fresh.initialize_from_storyboard(storyboard)
        for shot in storyboard.get("shots", []) or []:
            fresh.prepare_shot(shot)
            fresh.apply_shot(shot)
        fresh.report["summary"] = {
            "shots_total": len(storyboard.get("shots", []) or []),
            "conflicts": len(fresh.report["conflicts"]),
            "corrections": len(fresh.report["corrections"]),
            "reverse_shots": len(fresh.report["reverse_shots"]),
        }
        return fresh.report

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _record(self, category: str, shot: Dict[str, Any], kind: str, detail: str, corrected: bool) -> None:
        entry = {
            "shot_id": shot.get("shot_id", "unknown"),
            "scene_id": shot.get("scene_id", "scene_01"),
            "type": kind,
            "detail": detail,
            "corrected": corrected,
        }
        self.report.setdefault(category, []).append(entry)
        if category == "conflicts":
            log.warning("空间冲突 [%s] %s", shot.get("shot_id"), detail)
        else:
            log.info("空间连续性 [%s] %s", shot.get("shot_id"), detail)

    def _detect_camera_side(self, shot: Dict[str, Any], default: str = "A") -> str:
        side = shot.get("camera_side")
        if side in ("A", "B"):
            return side
        camera_text = str(shot.get("camera") or "")
        if "反打" in camera_text or ("对面" in camera_text and "180" in camera_text):
            return "B"
        return default

    def _is_explicit_reverse(self, shot: Dict[str, Any]) -> bool:
        if shot.get("camera_side") == "B":
            return True
        camera_text = str(shot.get("camera") or "")
        return "反打" in camera_text or ("对面" in camera_text and "180" in camera_text)

    def _expected_screen_positions(self, layout: Dict[str, Any], camera_side: str) -> Dict[str, str]:
        fixtures = layout.get("fixtures", {})
        if camera_side == "A":
            return {k: v for k, v in fixtures.items()}
        return {k: flip_horizontal(v) for k, v in fixtures.items()}

    def _infer_character_state(self, shot: Dict[str, Any], char: str) -> Dict[str, Any]:
        anchors = shot.get("spatial_anchors") or {}
        return {
            "scene_id": shot.get("scene_id", "scene_01"),
            "position": anchors.get("character_position") or anchors.get(f"{char}_position") or "center",
            "facing": anchors.get("character_facing") or anchors.get(f"{char}_facing") or "right",
            "motion_direction": anchors.get("motion_direction") or anchors.get(f"{char}_motion_direction") or "static",
        }

    def _validate_character_motion(
        self,
        shot: Dict[str, Any],
        char: str,
        scene_id: str,
        camera_side: str,
        reverse: bool,
        explicit_reverse: bool,
    ) -> None:
        anchors = shot.setdefault("spatial_anchors", {})
        facing = anchors.get("character_facing") or anchors.get(f"{char}_facing")
        motion = anchors.get("motion_direction") or anchors.get(f"{char}_motion_direction")
        prev = self.character_states.get(char)

        # 首次出现的角色：初始化
        if not prev or prev.get("scene_id") != scene_id:
            self.character_states[char] = {
                "scene_id": scene_id,
                "position": anchors.get("character_position") or "center",
                "facing": facing or "right",
                "motion_direction": motion or "static",
            }
            return

        if reverse and explicit_reverse:
            # 反打镜头：屏幕方向允许翻转，但必须已经显式标注
            if motion and prev.get("motion_direction") and motion == prev["motion_direction"]:
                self._record(
                    "reverse_shots", shot,
                    "motion_should_flip_in_reverse_shot",
                    f"反打镜头下角色 {char} 的屏幕运动方向应相对上一镜头翻转"
                    f"（上一镜头 {prev['motion_direction']}），当前锚点未翻转，请检查。",
                    corrected=False,
                )
            return

        # 同机位：方向锁
        if prev.get("motion_direction") and motion and prev["motion_direction"] != motion:
            if self._is_explicit_turn(shot):
                self._record(
                    "reverse_shots", shot,
                    "explicit_turn_annotated",
                    f"角色 {char} 运动方向由 {prev['motion_direction']} 变为 {motion}，"
                    f"已显式说明（如掉头/折返），允许。",
                    corrected=False,
                )
            else:
                self._record(
                    "conflicts", shot,
                    "motion_direction_break",
                    f"角色 {char} 上一镜头运动方向为 {prev['motion_direction']}，"
                    f"本镜头为 {motion}，未显式说明掉头/折返，已修正为 {prev['motion_direction']}。",
                    corrected=True,
                )
                anchors["motion_direction"] = prev["motion_direction"]
                if prev["motion_direction"] == "left_to_right":
                    anchors["character_facing"] = "right"
                elif prev["motion_direction"] == "right_to_left":
                    anchors["character_facing"] = "left"

        if prev.get("facing") and facing and prev["facing"] != facing:
            if not self._is_explicit_turn(shot):
                self._record(
                    "corrections", shot,
                    "facing_continuity",
                    f"角色 {char} 朝向由 {prev['facing']} 变为 {facing}，未显式说明，已修正为 {prev['facing']}。",
                    corrected=True,
                )
                anchors["character_facing"] = prev["facing"]

    def _is_explicit_turn(self, shot: Dict[str, Any]) -> bool:
        text = " ".join([
            str(shot.get("action") or ""),
            str(shot.get("camera") or ""),
            str(shot.get("narration") or ""),
            str(shot.get("motion_change_reason") or ""),
        ])
        return any(word in text for word in ("掉头", "折返", "转身往回", "turn around", "回头"))

    def _build_continuity_input(self, shot: Dict[str, Any], scene_id: str, prev_camera: str) -> Dict[str, Any]:
        inherited = {
            "camera_side": prev_camera,
            "scene_id": scene_id,
            "inherited_from": self.shot_history[-1].get("shot_id") if self.shot_history else None,
            "fixtures": copy.deepcopy(self.scene_layouts[scene_id]["fixtures"]),
            "characters": {},
        }
        for char in shot.get("characters", []) or []:
            if char in self.character_states:
                inherited["characters"][char] = copy.deepcopy(self.character_states[char])
        return inherited

    @staticmethod
    def _cn(value: Any) -> str:
        """把英文空间锚点翻译成中文，便于 MiniMax 文本编码器理解。"""
        if not isinstance(value, str):
            return str(value)
        v = value.lower().strip()
        table = {
            "left": "左侧",
            "right": "右侧",
            "center": "中央",
            "front": "前方",
            "back": "后方",
            "left_front": "左前方",
            "right_front": "右前方",
            "left_back": "左后方",
            "right_back": "右后方",
            "front_left": "前方偏左",
            "front_right": "前方偏右",
            "back_left": "后方偏左",
            "back_right": "后方偏右",
            "left_to_right": "从左到右",
            "right_to_left": "从右到左",
            "toward_camera": "朝向镜头",
            "away_from_camera": "远离镜头",
            "static": "静止",
            "a": "A",
            "b": "B",
        }
        return table.get(v, value)

    def snapshot(self) -> Dict[str, Any]:
        """当前完整状态快照。"""
        return {
            "scene_layouts": copy.deepcopy(self.scene_layouts),
            "character_states": copy.deepcopy(self.character_states),
            "camera_state": copy.deepcopy(self.camera_state),
        }
