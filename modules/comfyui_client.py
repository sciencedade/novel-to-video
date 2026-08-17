"""ComfyUI API 调用器（MiniMax H3）。

通过 /object_info 动态发现 MiniMax 相关节点，不硬编码不存在的节点名；
使用可配置节点映射将 workflow 模板中的角色占位符替换为实际 class_type。
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)


class ComfyUIError(RuntimeError):
    """ComfyUI 交互错误。"""


class ComfyUITimeout(ComfyUIError):
    """等待 ComfyUI 任务完成超时。"""


class ComfyUIClient:
    """ComfyUI HTTP 客户端。

    节点映射角色（role）：
    - MODEL_LOADER: MiniMax 模型加载节点
    - TEXT_TO_VIDEO: 文生视频节点
    - IMAGE_TO_VIDEO: 图生视频节点（首帧/尾帧输入）
    - SAVE_VIDEO: 视频保存/合成节点（VHS_VideoCombine / SaveVideo 等）
    - TEXT_TO_IMAGE: 首帧图像生成节点（可选，Flux/SDXL 等）
    - SAVE_IMAGE: 图像保存节点（可选）
    """

    def __init__(self, config: Dict[str, Any]):
        from .config_utils import PROJECT_ROOT, resolve_path

        # 合并顶层 comfyui 与 minimax_h3.comfyui；后者优先级更高（README 承诺的语义）
        top_cfg = config.get("comfyui", {}) or {}
        mh3_cfg = (config.get("minimax_h3") or {}).get("comfyui") or {}
        cfg = {**top_cfg, **mh3_cfg}
        top_mapping = {k.upper(): v for k, v in (top_cfg.get("node_mapping") or {}).items()}
        mh3_mapping = {k.upper(): v for k, v in (mh3_cfg.get("node_mapping") or {}).items()}
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:8188")).rstrip("/")
        self.poll_interval = float(cfg.get("poll_interval_seconds", 2))
        self.timeout = float(cfg.get("timeout_seconds", 7200))
        self.client_id = str(cfg.get("client_id", "novel2video"))
        self.workflow_template_path = resolve_path(cfg.get(
            "workflow_template", "workflow_templates/minimax_h3_workflow.json"))
        self.workflow_dir = resolve_path(cfg.get("workflow_dir", "workflow_templates"))
        _ff = cfg.get("first_frame_workflow", "")
        self.first_frame_workflow_path = resolve_path(_ff) if _ff else None
        # minimax_h3.comfyui.node_mapping 覆盖顶层 comfyui.node_mapping
        self.config_node_mapping = {**top_mapping, **mh3_mapping}
        self.input_name_overrides = cfg.get("input_name_overrides") or {}
        self.minimax_cfg = config.get("minimax", {}) or {}
        self.scan_cache_path = PROJECT_ROOT / "cache" / "comfyui_scan_cache.json"
        self.session = requests.Session()
        self._object_info: Optional[Dict[str, Any]] = None
        self.node_mapping: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 基础 HTTP
    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ping(self) -> bool:
        try:
            r = self.session.get(self._url("/system_stats"), timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_object_info(self, refresh: bool = False) -> Dict[str, Any]:
        """获取 /object_info 节点信息（带缓存）。"""
        if self._object_info is not None and not refresh:
            return self._object_info
        try:
            r = self.session.get(self._url("/object_info"), timeout=20)
            r.raise_for_status()
            self._object_info = r.json()
            log.info("已从 %s/object_info 获取 %d 个节点定义",
                     self.base_url, len(self._object_info))
            return self._object_info
        except Exception as exc:
            raise ComfyUIError(f"无法访问 ComfyUI /object_info（{self.base_url}）：{exc}") from exc

    # ------------------------------------------------------------------
    # 节点发现与映射
    # ------------------------------------------------------------------
    def discover_minimax_nodes(self) -> Dict[str, str]:
        """从 /object_info 自动发现 MiniMax 相关节点。

        优先选择直接输出 VIDEO 的轻量节点（如 MinimaxTextToVideoNode /
        MinimaxImageToVideoNode），避免需要 CLIP/VAE 的复杂扩散节点。
        """
        info = self.get_object_info()
        classes = list(info.keys())
        mini_classes = [c for c in classes if "minimax" in c.lower() or "h3" in c.lower()]

        def output_has(c: str, kind: str) -> bool:
            outputs = [str(o).lower() for o in info.get(c, {}).get("output", [])]
            return any(kind in o for o in outputs)

        def find_in(pool: List[str], predicates: List[str], prefer_video: bool = False,
                    prefer_node_suffix: bool = False, prefer_output: Optional[str] = None,
                    exclude: Optional[List[str]] = None) -> Optional[str]:
            best: Optional[str] = None
            best_score = -1
            for c in pool:
                low = c.lower()
                if exclude and any(e in low for e in exclude):
                    continue
                score = sum(1 for p in predicates if p in low)
                if score != len(predicates):
                    continue
                if prefer_video and output_has(c, "video"):
                    score += 2
                if prefer_output and output_has(c, prefer_output):
                    score += 2
                if prefer_node_suffix and low.endswith("node"):
                    score += 1
                if best_score < score:
                    best_score = score
                    best = c
            return best

        discovered: Dict[str, str] = {}
        discovered["MODEL_LOADER"] = (
            find_in(mini_classes, ["loader"]) or find_in(mini_classes, ["model", "load"]))
        discovered["TEXT_TO_VIDEO"] = (
            find_in(mini_classes, ["text", "video"], prefer_video=True, prefer_node_suffix=True)
            or find_in(mini_classes, ["t2v"], prefer_video=True))
        discovered["IMAGE_TO_VIDEO"] = (
            find_in(mini_classes, ["image", "video"], prefer_video=True, prefer_node_suffix=True)
            or find_in(mini_classes, ["i2v"], prefer_video=True))
        discovered["SAVE_VIDEO"] = (
            find_in(classes, ["save", "video"], exclude=["dataset"])
            or find_in(classes, ["vhs_videocombine"])
            or find_in(classes, ["video", "combine"]))
        discovered["TEXT_TO_IMAGE"] = (
            find_in(classes, ["text", "to", "image"], prefer_output="image",
                    prefer_node_suffix=True, exclude=["save", "load", "dataset", "encode"])
            or find_in(classes, ["text", "image"], prefer_output="image",
                       prefer_node_suffix=True, exclude=["save", "load", "dataset", "encode"]))
        discovered["SAVE_IMAGE"] = find_in(classes, ["save", "image"],
                                           exclude=["text", "dataset"])

        log.info("MiniMax 节点自动发现结果：%s", json.dumps(discovered, ensure_ascii=False))
        return discovered

    def build_node_mapping(self, refresh: bool = False) -> Dict[str, str]:
        """合并自动发现结果与用户配置覆盖。"""
        discovered = self.discover_minimax_nodes()
        mapping: Dict[str, str] = {}
        for role, cls in discovered.items():
            if cls:
                mapping[role] = cls
        for role, cls in self.config_node_mapping.items():
            if cls:
                mapping[role] = cls
        self.node_mapping = mapping
        self._validate_mapping(mapping)
        return mapping

    def _validate_mapping(self, mapping: Dict[str, str]) -> None:
        info = self.get_object_info()
        for role, cls in mapping.items():
            if cls and cls not in info:
                log.warning("节点映射 %s -> %s 在 /object_info 中不存在，请检查 config.yaml。", role, cls)
        if "TEXT_TO_VIDEO" not in mapping and "IMAGE_TO_VIDEO" not in mapping:
            raise ComfyUIError(
                "未在 ComfyUI 中发现 TEXT_TO_VIDEO / IMAGE_TO_VIDEO 节点，且 config.yaml 未配置 "
                "comfyui.node_mapping。请先安装 MiniMax H3 节点，或手动配置实际节点 class_type。")
        if "MODEL_LOADER" not in mapping:
            log.warning("未发现独立 MODEL_LOADER 节点；将使用视频节点内置的 model 下拉选项（COMBO）。")
        if "SAVE_VIDEO" not in mapping:
            log.warning("未发现 SAVE_VIDEO 节点；将依赖视频生成节点直接输出视频。")

    # ------------------------------------------------------------------
    # 环境扫描与缓存
    # ------------------------------------------------------------------
    def extract_minimax_models(self, info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """从 /object_info 中提取 MiniMax 节点的模型下拉选项。"""
        info = info or self.get_object_info()
        results: List[Dict[str, Any]] = []
        seen_models: set = set()
        for cls, spec in info.items():
            low = cls.lower()
            if "minimax" not in low and "h3" not in low and "hailuo" not in low:
                continue
            node_inputs = spec.get("input", {})
            for input_name in ("model", "model_name", "minimax_model"):
                entry = None
                for group in ("required", "optional"):
                    if input_name in node_inputs.get(group, {}):
                        entry = node_inputs[group][input_name]
                        break
                if entry is None:
                    continue
                type_name = str(entry[0]).upper() if isinstance(entry, list) and entry else ""
                if "COMBO" not in type_name:
                    continue
                options = self._extract_options(entry) or []
                for opt in options:
                    if opt not in seen_models:
                        seen_models.add(opt)
                        results.append({"node": cls, "input": input_name, "model": opt})
        return results

    def scan_environment(self) -> Dict[str, Any]:
        """扫描当前 ComfyUI 实例的 MiniMax 节点与模型，并写入缓存。"""
        info = self.get_object_info(refresh=True)
        mini_nodes: Dict[str, Dict[str, Any]] = {}
        for cls, spec in info.items():
            low = cls.lower()
            if "minimax" in low or "h3" in low or "hailuo" in low:
                node_inputs = spec.get("input", {})
                mini_nodes[cls] = {
                    "required": list(node_inputs.get("required", {}).keys()),
                    "optional": list(node_inputs.get("optional", {}).keys()),
                    "output": spec.get("output", []),
                }
        discovered = self.discover_minimax_nodes()
        models = self.extract_minimax_models(info)
        from .workflow_scanner import scan_workflow_files
        workflows = scan_workflow_files([self.workflow_dir])
        scan = {
            "base_url": self.base_url,
            "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nodes": mini_nodes,
            "discovered_mapping": discovered,
            "models": models,
            "workflows": workflows,
        }
        self.save_scan_cache(scan)
        return scan

    def save_scan_cache(self, scan: Dict[str, Any]) -> None:
        try:
            self.scan_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scan_cache_path, "w", encoding="utf-8") as f:
                json.dump(scan, f, ensure_ascii=False, indent=2)
            log.info("扫描结果已缓存：%s", self.scan_cache_path)
        except Exception as exc:
            log.warning("扫描结果缓存失败：%s", exc)

    def load_scan_cache(self) -> Optional[Dict[str, Any]]:
        try:
            if self.scan_cache_path.exists():
                with open(self.scan_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            log.warning("读取扫描缓存失败：%s", exc)
        return None

    # ------------------------------------------------------------------
    # 工作流模板
    # ------------------------------------------------------------------
    def load_workflow(self, path: Optional[str | Path] = None) -> Dict[str, Any]:
        from .workflow_scanner import load_workflow_json, to_api_format
        p = Path(path) if path else self.workflow_template_path
        if not p.is_absolute():
            from .config_utils import PROJECT_ROOT
            p = PROJECT_ROOT / p
        if not p.exists():
            raise ComfyUIError(f"工作流模板不存在：{p}")
        data = load_workflow_json(p)
        try:
            oi = self.get_object_info()
        except Exception:
            oi = None
        api = to_api_format(data, object_info=oi)
        log.info("已加载工作流：%s（%d 个节点，%s 格式）",
                 p, len(api), "UI→API" if data is not api else "API")
        return api

    def resolve_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """将模板中的 __ROLE__ 占位符替换为实际节点 class_type。

        MODEL_LOADER / SAVE_VIDEO / SAVE_IMAGE 缺失时移除对应节点（若视频节点自带
        model 下拉选项或直接输出视频，则不需要这些节点）。
        """
        mapping = self.node_mapping or self.build_node_mapping()
        removable_roles = {"MODEL_LOADER", "SAVE_VIDEO", "SAVE_IMAGE"}
        for node_id, node in list(workflow.items()):
            ct = node.get("class_type", "")
            if ct.startswith("__") and ct.endswith("__"):
                role = ct.strip("_").upper()
                cls = mapping.get(role)
                if cls:
                    node["class_type"] = cls
                elif role in removable_roles:
                    log.info("未发现 %s 节点，从工作流中移除节点 %s。", role, node_id)
                    workflow.pop(node_id)
                else:
                    raise ComfyUIError(
                        f"未找到节点映射：{role}。请在 config.yaml 的 comfyui.node_mapping 中"
                        f"配置实际节点 class_type（可通过 /object_info 查看）。")
        self._cleanup_dangling_links(workflow)
        return workflow

    def _cleanup_dangling_links(self, workflow: Dict[str, Any]) -> None:
        """移除指向已删除节点的输入链接（如 model: [\"1\", 0]）。"""
        existing_ids = set(workflow.keys())
        for node in workflow.values():
            for key, value in list(node.get("inputs", {}).items()):
                if (isinstance(value, list) and len(value) >= 2
                        and isinstance(value[0], str) and value[0] not in existing_ids):
                    log.debug("移除失效链接输入 %s.%s -> %s", node.get("class_type"), key, value[0])
                    node["inputs"].pop(key, None)

    # ------------------------------------------------------------------
    # 构建 MiniMax 视频工作流
    # ------------------------------------------------------------------
    def build_workflow(
        self,
        shot: Dict[str, Any],
        prompt: str,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """根据分镜镜头构建可直接提交 /prompt 的工作流。

        支持两种工作流来源：
        1. 内置角色占位模板（__TEXT_TO_VIDEO__ 等）——按 node_mapping 替换角色
        2. 用户选择的真实工作流（API 或 UI 格式）——自动识别工作流中的 MiniMax
           节点角色，并只填充识别到的节点（不替换 class_type）
        """
        from .workflow_scanner import detect_workflow_roles

        workflow = self.load_workflow()
        placeholder_mode = any(
            str(n.get("class_type", "")).startswith("__")
            for n in workflow.values()
        )

        if placeholder_mode:
            self.resolve_workflow(workflow)
            mapping = self.node_mapping
        else:
            # 真实工作流：识别工作流内部节点角色；配置的 node_mapping 若出现在
            # 工作流中则优先采用。
            self._cleanup_dangling_links(workflow)
            mapping = detect_workflow_roles(workflow, self.get_object_info())
            for role, cls in self.config_node_mapping.items():
                if cls and any(n.get("class_type") == cls for n in workflow.values()):
                    mapping[role] = cls
            log.info("用户工作流角色识别：%s", json.dumps(mapping, ensure_ascii=False))

        info = self.get_object_info()

        # 选择视频生成节点：有首帧图且工作流中存在 IMAGE_TO_VIDEO 时优先图生视频
        video_class = mapping.get("TEXT_TO_VIDEO")
        if start_image and mapping.get("IMAGE_TO_VIDEO"):
            video_class = mapping["IMAGE_TO_VIDEO"]

        duration = float(shot.get("duration_seconds", 5))
        fps = float(self.minimax_cfg.get("fps", 24))
        num_frames = max(1, int(round(duration * fps)))
        width, height, res_str = self._parse_resolution()
        negative = str(self.minimax_cfg.get("negative_prompt", ""))
        shot_id = str(shot.get("shot_id", "shot_000"))

        filled_video = False
        for node in workflow.values():
            ct = node.get("class_type", "")
            if placeholder_mode and ct in (mapping.get("TEXT_TO_VIDEO"),
                                           mapping.get("IMAGE_TO_VIDEO")):
                # 角色占位模板：可以把 __TEXT_TO_VIDEO__ 槽位替换为图生视频节点
                node["class_type"] = video_class
                self._fill_video_node(node, info, prompt=prompt, negative=negative,
                                      width=width, height=height, res_str=res_str,
                                      fps=fps, num_frames=num_frames, seed=seed,
                                      start_image=start_image, end_image=end_image)
                filled_video = True
            elif not placeholder_mode and ct == video_class:
                self._fill_video_node(node, info, prompt=prompt, negative=negative,
                                      width=width, height=height, res_str=res_str,
                                      fps=fps, num_frames=num_frames, seed=seed,
                                      start_image=start_image, end_image=end_image)
                filled_video = True
            elif ct == mapping.get("MODEL_LOADER"):
                self._fill_model_loader(node, info)
            elif mapping.get("SAVE_VIDEO") and ct == mapping["SAVE_VIDEO"]:
                self._fill_node_inputs(
                    node, {"filename_prefix": "video/" + shot_id}, info)

        if not filled_video:
            raise ComfyUIError(
                f"工作流中未找到可填充的视频生成节点（TEXT_TO_VIDEO={mapping.get('TEXT_TO_VIDEO')}, "
                f"IMAGE_TO_VIDEO={mapping.get('IMAGE_TO_VIDEO')}）。请检查工作流是否包含 MiniMax 视频生成节点。")
        self._fix_image_inputs(workflow, info)
        return workflow

    # ------------------------------------------------------------------
    # IMAGE 输入适配：start_image/end_image 等本地文件路径需上传到 ComfyUI
    # 并通过 LoadImage 节点转为张量（内置 MiniMaxH3ImageToVideo 的 first_frame
    # 等输入不接受字符串路径）。
    # ------------------------------------------------------------------
    def _fix_image_inputs(self, workflow: Dict[str, Any],
                          info: Dict[str, Any]) -> None:
        existing = [int(i) for i in workflow.keys() if str(i).isdigit()]
        next_id = (max(existing) + 1) if existing else 1
        pending = []  # (node, input_key, file_path)
        for node in workflow.values():
            ct = node.get("class_type", "")
            spec = info.get(ct, {}).get("input", {}) or {}
            all_in = {**spec.get("required", {}), **spec.get("optional", {})}
            for k, v in list(node.get("inputs", {}).items()):
                if not isinstance(v, str) or k not in all_in:
                    continue
                entry = all_in.get(k)
                type_name = entry[0] if (isinstance(entry, list) and entry
                                         and isinstance(entry[0], str)) else ""
                type_name = type_name.upper()
                if "IMAGE" not in type_name and "LATENT" not in type_name:
                    continue  # COMBO（选项列表）与普通字符串输入跳过
                pending.append((node, k, v))
        for node, k, v in pending:
            name = self._upload_image(v)
            load_id = str(next_id)
            next_id += 1
            workflow[load_id] = {"class_type": "LoadImage",
                                 "inputs": {"image": name}}
            node["inputs"][k] = [load_id, 0]
            log.info("首帧/尾帧已上传并接入 LoadImage(%s)：%s", load_id, name)

    def _upload_image(self, path: str) -> str:
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise ComfyUIError(f"参考图片不存在：{path}")
        suffix = p.suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(suffix.lstrip("."), "image/png")
        with open(p, "rb") as f:
            r = self.session.post(
                self._url("/upload/image"),
                files={"image": (p.name, f, mime)},
                data={"overwrite": "true", "type": "input"},
                timeout=120)
        r.raise_for_status()
        name = r.json().get("name") or p.name
        return name

    def _fill_model_loader(self, node: Dict[str, Any], info: Dict[str, Any]) -> None:
        spec = info.get(node.get("class_type", ""), {}).get("input", {})
        required = spec.get("required", {})
        model_input = None
        for candidate in ("model_name", "model", "minimax_model", "h3_model"):
            if candidate in required or candidate in spec.get("optional", {}):
                model_input = candidate
                break
        if not model_input:
            log.warning("模型加载节点 %s 未找到模型输入名，跳过填充。", node.get("class_type"))
            return
        configured = str(self.minimax_cfg.get("model_name") or "")
        if configured:
            node.setdefault("inputs", {})[model_input] = configured
        else:
            choices = required.get(model_input, [[]])[0] if model_input in required else []
            if isinstance(choices, list) and choices:
                node.setdefault("inputs", {})[model_input] = choices[0]
                log.info("未配置 minimax.model_name，自动选择第一个可用模型：%s", choices[0])
            else:
                log.warning("无法自动选择模型，请在 config.yaml 设置 minimax.model_name。")

    def _fill_video_node(
        self,
        node: Dict[str, Any],
        info: Dict[str, Any],
        prompt: str,
        negative: str,
        width: Optional[int],
        height: Optional[int],
        res_str: str,
        fps: float,
        num_frames: int,
        seed: Optional[int],
        start_image: Optional[str],
        end_image: Optional[str],
    ) -> None:
        class_type = node.get("class_type", "")
        spec = info.get(class_type, {}).get("input", {})
        all_inputs = {**spec.get("required", {}), **spec.get("optional", {})}

        values: Dict[str, Any] = {}
        # 提示词：不同节点包可能是 prompt_text / prompt / text / positive_prompt
        for key in ("prompt_text", "prompt", "text", "positive_prompt", "positive"):
            if key in all_inputs:
                values[key] = prompt
                break
        for key in ("negative_prompt", "negative"):
            if key in all_inputs:
                values[key] = negative
                break

        # 分辨率：支持 "1280x720" 字符串，也支持 COMBO（768P/1080p）选项
        self._fill_resolution(node, info, values, width, height, res_str)

        for key in ("fps", "frame_rate"):
            if key in all_inputs:
                values[key] = fps
                break

        # 帧数类输入（length/num_frames 等）与秒数类输入（duration 下拉）分别处理
        for key in ("num_frames", "length", "frame_count", "frames", "video_length"):
            if key in all_inputs:
                values[key] = num_frames
                break
        self._fill_duration(node, info, values, num_frames, fps)

        if seed is not None:
            for key in ("seed", "noise_seed"):
                if key in all_inputs:
                    values[key] = seed
                    break

        # 首尾帧：不同节点输入名不同
        if start_image:
            for key in ("start_image", "first_frame", "image", "first_frame_image", "init_image"):
                if key in all_inputs:
                    values[key] = start_image
                    break
        if end_image:
            for key in ("end_image", "last_frame", "last_frame_image", "last_image"):
                if key in all_inputs:
                    values[key] = end_image
                    break

        # model 输入：若模板中的 model 链接已被移除（无独立 loader），则填充 COMBO 选项
        self._fill_model_combo(node, info, values)

        self._fill_node_inputs(node, values, info)

    # ------------------------------------------------------------------
    # 输入填充辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _input_entry(info: Dict[str, Any], class_type: str, input_name: str):
        spec = info.get(class_type, {}).get("input", {})
        for group in ("required", "optional"):
            entry = spec.get(group, {}).get(input_name)
            if entry is not None:
                return entry
        return None

    @classmethod
    def _extract_options(cls, entry) -> Optional[List[str]]:
        if not isinstance(entry, list) or len(entry) < 2:
            return None
        meta = entry[1] if isinstance(entry[1], dict) else {}
        opts = meta.get("options")
        if not isinstance(opts, list):
            return None
        out: List[str] = []
        for o in opts:
            if isinstance(o, str):
                out.append(o)
            elif isinstance(o, dict) and o.get("key"):
                out.append(str(o["key"]))
        return out or None

    def _fill_model_combo(self, node: Dict[str, Any], info: Dict[str, Any],
                          values: Dict[str, Any]) -> None:
        class_type = node.get("class_type", "")
        existing = node.get("inputs", {}).get("model")
        if isinstance(existing, list):  # 保留节点间链接
            return
        entry = self._input_entry(info, class_type, "model")
        if entry is None:
            return
        type_name = str(entry[0]).upper()
        if "COMBO" not in type_name:
            return
        options = self._extract_options(entry) or []
        configured = str(self.minimax_cfg.get("model_name") or "")
        if configured and (not options or configured in options):
            values["model"] = configured
            log.info("使用配置的 MiniMax 模型：%s", configured)
        elif existing not in (None, ""):
            # 用户工作流中已选择模型且未显式配置时，保留工作流原有选择
            log.info("保留工作流中已有的模型选择：%s", existing)
        elif options:
            values["model"] = options[0]
            log.info("未配置 minimax.model_name，自动选择第一个模型：%s", options[0])

    def _fill_duration(self, node: Dict[str, Any], info: Dict[str, Any],
                       values: Dict[str, Any], num_frames: int, fps: float) -> None:
        class_type = node.get("class_type", "")
        entry = self._input_entry(info, class_type, "duration")
        if entry is None:
            return
        duration_seconds = num_frames / max(fps, 1)
        type_name = str(entry[0]).upper()
        if "COMBO" in type_name:
            options = self._extract_options(entry) or []
            if options:
                numeric = []
                for o in options:
                    try:
                        numeric.append(float(o))
                    except ValueError:
                        continue
                if numeric:
                    values["duration"] = min(numeric, key=lambda x: abs(x - duration_seconds))
                else:
                    values["duration"] = options[0]
        else:
            values["duration"] = int(round(duration_seconds))

    def _fill_resolution(self, node: Dict[str, Any], info: Dict[str, Any],
                          values: Dict[str, Any], width: Optional[int],
                          height: Optional[int], res_str: str) -> None:
        class_type = node.get("class_type", "")
        entry = self._input_entry(info, class_type, "resolution")
        if entry is None:
            # 无 resolution 输入则填 width/height
            spec = info.get(class_type, {}).get("input", {})
            all_inputs = {**spec.get("required", {}), **spec.get("optional", {})}
            for key in ("width", "image_width"):
                if key in all_inputs:
                    values[key] = width
                    break
            for key in ("height", "image_height"):
                if key in all_inputs:
                    values[key] = height
                    break
            return
        type_name = str(entry[0]).upper()
        if "COMBO" in type_name:
            options = self._extract_options(entry) or []
            if res_str in options:
                values["resolution"] = res_str
            elif options:
                # 按目标宽度选择最接近的档位（如 768P / 1080p / 2K）
                target = width or 1280
                def rank(o: str) -> float:
                    digits = re.findall(r'\d+', o)
                    if not digits:
                        return 9999
                    return abs(int(digits[0]) - target)
                values["resolution"] = sorted(options, key=rank)[0]
        else:
            values["resolution"] = res_str

    def _fill_node_inputs(self, node: Dict[str, Any], values: Dict[str, Any],
                          info: Dict[str, Any]) -> None:
        spec = info.get(node.get("class_type", ""), {}).get("input", {})
        all_inputs = {**spec.get("required", {}), **spec.get("optional", {})}
        inputs = node.setdefault("inputs", {})
        for key, value in values.items():
            if key in all_inputs:
                inputs[key] = value
            else:
                log.debug("节点 %s 没有输入 %s，忽略。", node.get("class_type"), key)
        # 清理未解析的占位符
        for key in list(inputs.keys()):
            v = inputs[key]
            if isinstance(v, str) and v.startswith("__") and v.endswith("__"):
                inputs.pop(key)
                log.debug("移除未解析占位输入 %s.%s", node.get("class_type"), key)

    def cancel_current(self) -> None:
        """请求 ComfyUI 中断当前任务（重试前调用，避免任务堆积）。"""
        try:
            r = self.session.post(self._url("/interrupt"), json={}, timeout=10)
            if r.status_code < 400:
                log.warning("已请求 ComfyUI /interrupt 取消当前任务。")
            else:
                log.warning("ComfyUI /interrupt 返回 HTTP %s：%s",
                            r.status_code, r.text[:200])
        except Exception as exc:
            log.warning("请求 ComfyUI /interrupt 失败：%s", exc)

    def _parse_resolution(self) -> Tuple[Optional[int], Optional[int], str]:
        res = str(self.minimax_cfg.get("resolution", "1280x720"))
        m = re.match(r'(\d+)\s*[xX×]\s*(\d+)', res)
        if m:
            return int(m.group(1)), int(m.group(2)), res
        return None, None, res

    # ------------------------------------------------------------------
    # 提交与轮询
    # ------------------------------------------------------------------
    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            r = self.session.post(self._url("/prompt"), json=payload, timeout=30)
        except Exception as exc:
            raise ComfyUIError(f"提交 /prompt 失败：{exc}") from exc
        if r.status_code >= 400:
            raise ComfyUIError(
                f"提交 /prompt 失败 HTTP {r.status_code}：{r.text[:1000]}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI /prompt 返回异常：{data}")
        log.info("已提交 ComfyUI 任务，prompt_id=%s", prompt_id)
        return prompt_id

    def wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                r = self.session.get(self._url(f"/history/{prompt_id}"), timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                log.warning("轮询 /history 失败：%s，继续等待…", exc)
                time.sleep(self.poll_interval)
                continue

            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(
                        f"ComfyUI 任务执行错误：{json.dumps(status, ensure_ascii=False)[:500]}")
                if entry.get("outputs"):
                    return entry
                if status.get("completed"):
                    return entry
            time.sleep(self.poll_interval)
        raise ComfyUITimeout(f"等待 ComfyUI 任务 {prompt_id} 完成超时（{self.timeout}s）")

    def get_history_entry(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = self.session.get(self._url(f"/history/{prompt_id}"), timeout=30)
            r.raise_for_status()
            return r.json().get(prompt_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 输出下载
    # ------------------------------------------------------------------
    @staticmethod
    def extract_output_files(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        seen: set = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ("images", "videos", "gifs") and isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and item.get("filename"):
                                sig = (item.get("filename"), item.get("subfolder"), item.get("type"))
                                if sig not in seen:
                                    seen.add(sig)
                                    files.append({"kind": key, **item})
                    else:
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(entry.get("outputs", {}))
        video_exts = {".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv"}
        videos = [f for f in files if f.get("kind") in ("videos", "gifs")
                  or Path(f.get("filename", "")).suffix.lower() in video_exts]
        images = [f for f in files if f not in videos]
        return videos + images

    def download_file(self, file_info: Dict[str, Any], dest: str | Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "filename": file_info.get("filename"),
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
        r = self.session.get(self._url("/view"), params=params, timeout=300, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "json" in content_type:
            raise ComfyUIError(f"下载输出失败：{r.text[:300]}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        log.info("已下载输出：%s", dest)
        return dest

    def download_output(self, entry: Dict[str, Any], dest_dir: str | Path,
                        shot_id: str) -> Optional[Path]:
        files = self.extract_output_files(entry)
        if not files:
            return None
        target = files[0]
        ext = Path(target.get("filename", ".mp4")).suffix or ".mp4"
        dest = Path(dest_dir) / f"{shot_id}{ext}"
        return self.download_file(target, dest)

    # ------------------------------------------------------------------
    # 首帧图像生成（可选）
    # ------------------------------------------------------------------
    def generate_image(self, prompt: str, prefix: str,
                       dest_dir: str | Path = "shots/frames",
                       workflow_path: Optional[str | Path] = None,
                       width: Optional[int] = None,
                       height: Optional[int] = None,
                       seed: Optional[int] = None,
                       resolution: Optional[str] = None) -> Optional[str]:
        """通过 ComfyUI 文生图工作流生成一张图并下载到本地。

        参数：
        - prompt: 图像提示词
        - prefix: 输出文件前缀
        - dest_dir: 下载目录
        - workflow_path: 文生图工作流 JSON（默认 first_frame_workflow）
        - width/height/resolution: 图像尺寸；缺省用 minimax.resolution
        - seed: 随机种子
        """
        wf_path = Path(workflow_path) if workflow_path else self.first_frame_workflow_path
        if not wf_path or not wf_path.exists():
            log.info("未配置图像生成工作流：%s", wf_path)
            return None
        try:
            workflow = self.load_workflow(wf_path)
            self.resolve_workflow(workflow)
        except Exception as exc:
            log.warning("图像生成工作流不可用：%s", exc)
            return None

        info = self.get_object_info()
        dw, dh, dres = self._parse_resolution()
        if width is None:
            width = dw
        if height is None:
            height = dh
        if resolution is None:
            resolution = dres
        mapping = self.node_mapping
        for node in workflow.values():
            ct = node.get("class_type", "")
            if mapping.get("TEXT_TO_IMAGE") and ct == mapping["TEXT_TO_IMAGE"]:
                values: Dict[str, Any] = {}
                all_inputs = {**info.get(ct, {}).get("input", {}).get("required", {}),
                              **info.get(ct, {}).get("input", {}).get("optional", {})}
                for key in ("prompt", "text", "positive_prompt", "positive"):
                    if key in all_inputs:
                        values[key] = prompt
                        break
                if "resolution" in all_inputs:
                    values["resolution"] = resolution
                else:
                    for key in ("width", "image_width"):
                        if key in all_inputs:
                            values[key] = width
                            break
                    for key in ("height", "image_height"):
                        if key in all_inputs:
                            values[key] = height
                            break
                if seed is not None:
                    for key in ("seed", "noise_seed"):
                        if key in all_inputs:
                            values[key] = seed
                            break
                self._fill_node_inputs(node, values, info)
            elif mapping.get("SAVE_IMAGE") and ct == mapping["SAVE_IMAGE"]:
                self._fill_node_inputs(node, {"filename_prefix": prefix}, info)

        try:
            prompt_id = self.submit_workflow(workflow)
            entry = self.wait_for_completion(prompt_id)
            path = self.download_output(entry, Path(dest_dir), prefix)
            return str(path) if path else None
        except Exception as exc:
            log.warning("图像生成失败：%s", exc)
            return None

    def generate_first_frame(self, prompt: str, shot_id: str,
                             dest_dir: str | Path = "shots/frames") -> Optional[str]:
        return self.generate_image(prompt, f"{shot_id}_first_frame",
                                   dest_dir=dest_dir,
                                   workflow_path=self.first_frame_workflow_path)
