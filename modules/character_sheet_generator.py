"""角色定妆照生成器。

通过 ComfyUI 文生图工作流为指定角色生成定妆照（正面/侧面/全身等），
保存到 assets/characters/<角色名>/ 下，供 character_references 使用。

支持两种工作流：
- 普通文生图模板（character_sheet_workflow.json）
- Mage-Flow + IPAdapter 模板（character_sheet_mage_flow_ipadapter.json）：
  通过 style_reference_image 把风格参考图交给 IPAdapter，统一所有角色定妆照的画风。
"""

from __future__ import annotations

import logging
import re
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
        self.workflow_path = str(
            cs.get("workflow_template")
            or "workflow_templates/character_sheet_mage_flow_t2i.json"
        )
        self.ipadapter_workflow_path = str(
            cs.get("ipadapter_workflow_template")
            or "workflow_templates/character_sheet_mage_flow_ipadapter.json"
        )
        self.style_reference_image = str(cs.get("style_reference_image") or "")
        self.resolution = str(cs.get("resolution", "1024x1024"))
        self.default_angles = list(cs.get("default_angles", ["front", "side", "full"]))
        self.prompt_template = str(
            cs.get(
                "prompt_template",
                "character design sheet, {name}, {angle} view, "
                "plain background, even soft lighting, high quality",
            )
        )
        self.negative_prompt = str(cs.get("negative_prompt") or "")
        self.unet_name = str(cs.get("unet_name") or "mage_flow_int8_convrot.safetensors")
        self.clip_name = str(cs.get("clip_name") or "qwen3vl_4b_bf16.safetensors")
        self.vae_name = str(cs.get("vae_name") or "mage_flow_vae_bf16.safetensors")
        self.ipadapter_file = str(
            cs.get("ipadapter_file") or "ip-adapter-faceid.sdxl.bin")
        try:
            self.ipadapter_weight = float(cs.get("ipadapter_weight", 0.6))
        except (TypeError, ValueError):
            self.ipadapter_weight = 0.6
        try:
            self.seed = int(config.get("minimax", {}).get("seed", -1))
        except (TypeError, ValueError):
            self.seed = -1
        self._width, self._height = self._parse_resolution(self.resolution)
        self._style_ref_name: Optional[str] = None

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

        # 有风格参考图时启用 IPAdapter 工作流；否则用普通文生图工作流
        if self.style_reference_image:
            if not Path(self.style_reference_image).exists():
                raise RuntimeError(
                    f"character_sheet.style_reference_image 不存在：{self.style_reference_image}"
                )
            self.workflow_path = self.ipadapter_workflow_path
            if self._style_ref_name is None:
                self._style_ref_name = self.client.upload_image(self.style_reference_image)
                log.info("风格参考图已上传：%s", self._style_ref_name)
        else:
            log.info("未配置 style_reference_image，使用普通文生图工作流（无 IPAdapter）。")

        results: List[Dict[str, str]] = []
        for angle in angles:
            angle_key = angle.lower().replace(" ", "_").replace("/", "_")
            p = self._build_prompt(name, prompt, angle)
            prefix = f"{name}_{angle_key}"
            log.info("生成定妆照：%s（%s 视角）", name, angle)
            try:
                path = self._generate_one(p, prefix, char_dir)
            except Exception as exc:
                log.warning("定妆照生成失败：%s（%s）: %s", name, angle, exc)
                path = None
            if path:
                results.append({"angle": angle, "path": path})
                log.info("定妆照已生成：%s", path)
        if not results:
            raise RuntimeError(
                f"角色 {name} 的定妆照全部生成失败。请检查 ComfyUI 图像生成节点/工作流。"
            )
        return results

    def _generate_one(self, prompt: str, prefix: str, char_dir: Path) -> Optional[str]:
        workflow = self.client.load_workflow(self.workflow_path)
        self.client.resolve_workflow(workflow)
        values: Dict[str, Any] = {
            "PROMPT": prompt,
            "NEGATIVE_PROMPT": self.negative_prompt,
            "WIDTH": self._width,
            "HEIGHT": self._height,
            "SEED": self.seed,
            "UNET_NAME": self.unet_name,
            "CLIP_NAME": self.clip_name,
            "VAE_NAME": self.vae_name,
            "IPADAPTER_FILE": self.ipadapter_file,
            "IPADAPTER_WEIGHT": self.ipadapter_weight,
            "STYLE_REFERENCE": self._style_ref_name or "",
            "PREFIX": prefix,
        }
        self.client.fill_placeholders(workflow, values)
        prompt_id = self.client.submit_workflow(workflow)
        entry = self.client.wait_for_completion(prompt_id)
        path = self.client.download_output(entry, char_dir, prefix)
        return str(path) if path else None

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

    @staticmethod
    def _parse_resolution(resolution: str):
        m = re.match(r'(\d+)\s*[xX×]\s*(\d+)', str(resolution))
        if m:
            return int(m.group(1)), int(m.group(2))
        return 1024, 1024
