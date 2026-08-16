"""配置加载与归一化工具。

统一处理三种 MiniMax H3 来源模式：
- local_comfyui: 本地 ComfyUI
- remote_comfyui: 远程 ComfyUI
- remote_api: MiniMax 官方/第三方 HTTP API

所有路径均以项目根目录为基准解析，不依赖当前电脑的绝对路径。
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _app_root() -> Path:
    """应用根目录。

    - 源码运行：项目根目录（config_utils.py 的上两级）
    - PyInstaller 冻结运行：exe 所在目录（可写，config/output/shots/logs 放这里）
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _resource_root() -> Path:
    """只读资源目录（打包进 exe 的 workflow_templates/examples/config.yaml 等）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else PROJECT_ROOT


PROJECT_ROOT = _app_root()
RESOURCE_ROOT = _resource_root()

DEFAULTS: Dict[str, Any] = {
    "comfyui": {
        "base_url": "http://127.0.0.1:8188",
        "poll_interval_seconds": 2,
        "timeout_seconds": 7200,
        "client_id": "novel2video",
        "workflow_template": "workflow_templates/minimax_h3_workflow.json",
        "workflow_dir": "workflow_templates",
        "first_frame_workflow": "workflow_templates/first_frame_workflow.json",
        "node_mapping": {},
        "input_name_overrides": {},
    },
    "minimax": {
        "model_name": "",
        "resolution": "1280x720",
        "fps": 24,
        "max_shot_duration": 10,
        "seed": -1,
        "prompt_style": "cinematic lighting, coherent film style, consistent character appearance",
        "negative_prompt": "blurry, distorted, morphing, flickering, inconsistent geometry, text artifacts",
    },
    "minimax_h3": {
        "provider": "local_comfyui",
        "model_name": "",
        "comfyui": {},
        "api": {
            "base_url": "https://api.minimaxi.com/v1",
            "api_key": "",
            "model": "MiniMax-H3",
            "create_path": "/video_generation",
            "query_path": "/query/video_generation",
            "timeout_seconds": 7200,
            "poll_interval_seconds": 3,
        },
    },
    "llm": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.4,
        "max_tokens": 64000,
        "seed": "",
        "timeout_seconds": 600,
    },
    "storyboard": {
        "mode": "auto",
        "segment_duration_seconds": 10,
        "chars_per_second": 5,
        "use_reference_image": True,
        "generate_first_frame": False,
        "first_frame_prompt": "",
    },
    "generation": {
        "auto_run": False,
        "concurrent_jobs": 1,
        "max_retries": 3,
        "retry_backoff_seconds": 5,
        "output_dir": "output",
        "shots_dir": "shots",
    },
    "ffmpeg": {
        "path": "",
        "transitions": False,
        "transition_duration": 0.5,
        "subtitles": False,
        "crf": 18,
        "preset": "medium",
    },
    "visual_qa": {
        "enabled": True,
        "resize": 64,
        "hash_size": 16,
        "drift_threshold": 0.85,
        "mirror_ratio_threshold": 1.15,
        "min_file_size_kb": 10,
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """加载 YAML 配置。

    相对路径按应用根目录（exe 目录/项目目录）解析。
    若默认 config.yaml 不存在，回退到打包资源中的模板。
    """
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        bundled = RESOURCE_ROOT / "config.yaml"
        if p.name == "config.yaml" and bundled.exists() and bundled != p:
            p = bundled
        else:
            raise FileNotFoundError(f"配置文件不存在：{p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误：{p}")
    return data


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把 minimax_h3 配置段归一化为运行时使用的 comfyui/minimax 段。

    - local_comfyui / remote_comfyui：minimax_h3.comfyui 合并进顶层 comfyui
    - remote_api：保留 minimax_h3.api 作为远程 API 客户端配置
    """
    cfg = copy.deepcopy(cfg)
    cfg = _deep_merge(copy.deepcopy(DEFAULTS), cfg)

    mh3 = cfg.get("minimax_h3") or {}
    provider = str(mh3.get("provider", "local_comfyui")).lower().strip()
    cfg["minimax_h3"]["provider"] = provider
    cfg["_provider"] = provider

    if provider in ("local_comfyui", "remote_comfyui"):
        comfy_override = mh3.get("comfyui") or {}
        if comfy_override:
            _deep_merge(cfg["comfyui"], comfy_override)
        # minimax_h3.model_name 优先于顶层 minimax.model_name
        if mh3.get("model_name"):
            cfg["minimax"]["model_name"] = mh3["model_name"]
        elif cfg["minimax"].get("model_name"):
            cfg["minimax_h3"]["model_name"] = cfg["minimax"]["model_name"]
    elif provider == "remote_api":
        api = mh3.get("api") or {}
        cfg["_remote_api"] = api
        if api.get("model"):
            cfg["minimax"]["model_name"] = api["model"]
    else:
        raise ValueError(
            f"未知的 minimax_h3.provider：{provider}。"
            f"可选值：local_comfyui / remote_comfyui / remote_api。")

    # 路径统一转为基于项目根目录的绝对路径（输出、shots、日志）
    for section, key in (("generation", "output_dir"), ("generation", "shots_dir"),
                         ("logging", "log_dir")):
        value = cfg.get(section, {}).get(key, "")
        if value:
            p = Path(value)
            if not p.is_absolute():
                cfg[section][key] = str(PROJECT_ROOT / p)
    return cfg


def detect_ffmpeg(cfg: Optional[Dict[str, Any]] = None) -> str:
    """返回可用的 ffmpeg 路径。配置优先，其次 PATH 自动检测。"""
    configured = ""
    if cfg:
        configured = str((cfg.get("ffmpeg") or {}).get("path") or "")
    if configured:
        return configured
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    return found or "ffmpeg"


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """把相对路径解析为绝对路径。

    查找顺序：指定 base → 应用根目录（exe 目录/项目目录）→ 打包资源目录（_MEIPASS）。
    这样打包后：用户可写文件在 exe 目录，只读模板/示例在资源目录。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    candidates: list[Path] = []
    if base is not None:
        candidates.append(base / p)
    candidates.append(PROJECT_ROOT / p)
    if RESOURCE_ROOT != PROJECT_ROOT:
        candidates.append(RESOURCE_ROOT / p)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else p
