"""角色定妆照生成器。

通过 ComfyUI 文生图工作流为指定角色生成定妆照（正面/侧面/全身等），
保存到 assets/characters/<角色名>/ 下，供 character_references 使用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_utils import resolve_path

log = logging.getLogger(__name__)


class CharacterSheetGenerator:
    """用 ComfyUI 文生图工作流生成角色定妆照。"""

    def __init__(self, config: Dict[str, Any], client=None):
        from .comfyui_client import ComfyUIClient

        self.config = config
        self.client = client or ComfyUIClient(config)
        cs = config.get("character_sheet") or {}
        self.output_dir = resolve_path(cs.get("output_dir", "assets/characters"))
        self.workflow_path = (
            cs.get("workflow_template")
            or config.get("comfyui", {}).get("first_frame_workflow")
            or "workflow_templates/first_frame_workflow.json"
        )
        self.resolution = str(cs.get("resolution", "1024x1024"))
        self.default_angles = list(cs.get("default_angles", ["front", "side", "full"]))
        self.prompt_template = str(
            cs.get(
                "prompt_template",
                "character design sheet, {name}, {angle} view, "
                "plain background, even soft lighting, high quality",
            )
        )
        try:
            self.seed = int(config.get("minimax", {}).get("seed", -1))
        except (TypeError, ValueError):
            self.seed = -1

    def generate(self, name: str, prompt: Optional[str] = None,
                 angles: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """生成一个角色的多角度定妆照。

        返回 [{"angle": "front", "path": "assets/characters/xxx/front.png"}, ...]
        """
        if not name or not str(name).strip():
            raise ValueError("角色名不能为空。")
        angles = [str(a).strip() for a in (angles or self.default_angles) if str(a).strip()]
        char_dir = self.output_dir / str(name).strip()
        char_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, str]] = []
        for angle in angles:
            angle_key = angle.lower().replace(" ", "_").replace("/", "_")
            p = self._build_prompt(name, prompt, angle)
            prefix = f"{name}_{angle_key}"
            log.info("生成定妆照：%s（%s 视角）", name, angle)
            path = self.client.generate_image(
                p,
                prefix,
                dest_dir=char_dir,
                workflow_path=self.workflow_path,
                resolution=self.resolution,
                seed=self.seed,
            )
            if path:
                results.append({"angle": angle, "path": str(path)})
                log.info("定妆照已生成：%s", path)
            else:
                log.warning("定妆照生成失败：%s（%s）", name, angle)
        if not results:
            raise RuntimeError(
                f"角色 {name} 的定妆照全部生成失败。请检查 ComfyUI 图像生成节点/工作流。"
            )
        return results

    def _build_prompt(self, name: str, prompt: Optional[str], angle: str) -> str:
        if prompt and str(prompt).strip():
            base = str(prompt).strip()
        else:
            base = self.prompt_template.format(name=name, angle=angle)
        # 保证角度描述在提示词里
        angle_word = str(angle).lower().strip()
        if angle_word and angle_word not in base.lower():
            base = f"{base}, {angle_word} view"
        return base
