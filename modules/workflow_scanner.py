"""ComfyUI 工作流扫描与接入模块。

支持两种工作流 JSON 格式：
1. API 格式（Save (API Format) 导出）：顶层为 {节点id: {class_type, inputs}}
2. UI 格式（普通 Save 导出）：{"nodes": [...], "links": [...], ...}
   UI 格式会自动转换为 API 格式。

扫描指定目录中的工作流 JSON，识别其中 MiniMax 相关节点，
并把角色（MODEL_LOADER/TEXT_TO_VIDEO/IMAGE_TO_VIDEO/SAVE_VIDEO）映射到
工作流中实际存在的节点 class_type，使用户可以直接选择自己的工作流接入。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def load_workflow_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"工作流文件格式错误（顶层不是对象）：{p}")
    return data


def is_ui_format(data: Dict[str, Any]) -> bool:
    """UI 格式包含 nodes 列表与 links 列表。"""
    return isinstance(data.get("nodes"), list)


def is_api_format(data: Dict[str, Any]) -> bool:
    """API 格式：至少有一个节点的 class_type 字段。"""
    if not data:
        return False
    return any(isinstance(v, dict) and "class_type" in v for v in data.values())


def ui_to_api(data: Dict[str, Any], object_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把 ComfyUI 界面导出格式转换为 API 格式。

    - 只把 widgets_values 填充给带 widget 标记的输入（可选图像输入如 first_frame
      无 widget 标记，保持缺省，避免错配）
    - 跳过前端节点（MarkdownNote 等）与后端不存在的节点
    """
    nodes = data.get("nodes") or []
    links = data.get("links") or []
    link_map: Dict[int, list] = {link[0]: link for link in links if isinstance(link, list) and link}

    api: Dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        class_type = node.get("type")
        if node_id is None or not class_type:
            continue
        if object_info is not None and class_type not in object_info:
            continue  # 前端节点/未安装插件：后端不可见，跳过
        new_id = str(node_id)
        inputs: Dict[str, Any] = {}
        widgets = node.get("widgets_values") or []
        widget_idx = 0
        for inp in node.get("inputs") or []:
            if not isinstance(inp, dict):
                continue
            name = inp.get("name")
            if not name:
                continue
            if name == "upload":
                continue  # LoadImage 的上传隐藏输入，API 格式不接受
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                link = link_map[link_id]
                # links: [link_id, from_node, from_slot, to_node, to_slot, type]
                inputs[name] = [str(link[1]), int(link[2])]
            elif inp.get("widget") is not None and widget_idx < len(widgets):
                # 只填充真正的 widget 输入；可选图像输入（first_frame 等）无 widget
                # 标记，跳过并保留缺省
                inputs[name] = widgets[widget_idx]
                widget_idx += 1
        api[new_id] = {"class_type": str(class_type), "inputs": inputs}
    return api


def to_api_format(data: Dict[str, Any], object_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """统一转换为 API 格式。"""
    if is_ui_format(data):
        return ui_to_api(data, object_info=object_info)
    if is_api_format(data):
        return data
    raise ValueError("无法识别的工作流格式：既不是 API 格式，也不是 UI 格式。")


def _output_has(info: Optional[Dict[str, Any]], cls: str, kind: str) -> bool:
    if not info:
        return False
    outputs = [str(o).lower() for o in info.get(cls, {}).get("output", [])]
    return any(kind in o for o in outputs)


def detect_workflow_roles(workflow: Dict[str, Any],
                          object_info: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """从 API 工作流中识别 MiniMax 相关节点的角色映射。

    返回 {ROLE: class_type}，只包含工作流中实际存在的节点。
    """
    classes = [str(node.get("class_type", "")) for node in workflow.values()
               if not str(node.get("class_type", "")).startswith("__")]
    mini = [c for c in classes if "minimax" in c.lower() or "h3" in c.lower()
            or "hailuo" in c.lower()]

    def find_in(pool: List[str], predicates: List[str], prefer_video: bool = False,
                prefer_node_suffix: bool = False, exclude: Optional[List[str]] = None) -> Optional[str]:
        best: Optional[str] = None
        best_score = -1
        for c in pool:
            low = c.lower()
            if exclude and any(e in low for e in exclude):
                continue
            score = sum(1 for p in predicates if p in low)
            if score != len(predicates):
                continue
            if prefer_video and _output_has(object_info, c, "video"):
                score += 2
            if prefer_node_suffix and low.endswith("node"):
                score += 1
            if best_score < score:
                best_score = score
                best = c
        return best

    roles: Dict[str, str] = {}
    roles["MODEL_LOADER"] = (
        find_in(mini, ["loader"]) or find_in(mini, ["model", "load"]))
    roles["TEXT_TO_VIDEO"] = (
        find_in(mini, ["text", "video"], prefer_video=True, prefer_node_suffix=True)
        or find_in(mini, ["t2v"], prefer_video=True))
    roles["IMAGE_TO_VIDEO"] = (
        find_in(mini, ["image", "video"], prefer_video=True, prefer_node_suffix=True)
        or find_in(mini, ["i2v"], prefer_video=True))
    roles["SAVE_VIDEO"] = (
        find_in(classes, ["save", "video"], exclude=["dataset"])
        or find_in(classes, ["vhs_videocombine"])
        or find_in(classes, ["video", "combine"]))
    return {k: v for k, v in roles.items() if v}


def scan_workflow_files(dirs: List[str | Path]) -> List[Dict[str, Any]]:
    """扫描目录中的工作流 JSON 文件，返回可接入的工作流列表。"""
    results: List[Dict[str, Any]] = []
    seen: set = set()
    for d in dirs:
        base = Path(d)
        if not base.exists():
            continue
        for p in sorted(base.glob("*.json")):
            try:
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                data = load_workflow_json(p)
                fmt = "ui" if is_ui_format(data) else ("api" if is_api_format(data) else "unknown")
                if fmt == "unknown":
                    continue
                api = to_api_format(data)
                roles = detect_workflow_roles(api)
                mini_nodes = sorted({str(n.get("class_type")) for n in api.values()
                                     if "minimax" in str(n.get("class_type", "")).lower()
                                     or "h3" in str(n.get("class_type", "")).lower()
                                     or "hailuo" in str(n.get("class_type", "")).lower()})
                results.append({
                    "path": str(p),
                    "name": p.name,
                    "format": fmt,
                    "node_count": len(api),
                    "mini_nodes": mini_nodes,
                    "roles": roles,
                })
            except Exception as exc:
                log.warning("跳过无法解析的工作流 %s：%s", p, exc)
    return results
