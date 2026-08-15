"""首次运行配置向导（交互式命令行）。

引导用户选择 MiniMax H3 来源模式（本地 ComfyUI / 远程 ComfyUI / 远程 API），
自动扫描 ComfyUI 节点与模型，检测 ffmpeg，并生成 config.yaml。

用法：
    python wizard.py
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.config_utils import DEFAULTS, PROJECT_ROOT, detect_ffmpeg, load_config, resolve_path
from modules.comfyui_client import ComfyUIClient
from modules.workflow_scanner import scan_workflow_files

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def ask(prompt: str, default: str = "", allow_empty: bool = False) -> str:
    suffix = f" [{default}]" if default else (" (可留空)" if allow_empty else "")
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(1)
        if value:
            return value
        if default:
            return default
        if allow_empty:
            return ""
        print("  该项不能为空，请重新输入。")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"{prompt} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        sys.exit(1)
    if not value:
        return default
    return value in ("y", "yes", "是")


def choose(options: List[str], prompt: str = "请选择") -> Optional[int]:
    if not options:
        return None
    print(f"{prompt}：")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            raw = input("  输入序号（回车=第 1 项）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(1)
        if not raw:
            return 0
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  序号无效，请重新输入。")


def scan_comfyui(base_url: str) -> Optional[Dict[str, Any]]:
    """尝试扫描 ComfyUI 中的 MiniMax 节点与模型。"""
    print(f"\n正在扫描 {base_url} 的 /object_info …")
    try:
        client = ComfyUIClient({"comfyui": {"base_url": base_url}, "minimax": {}})
        return client.scan_environment()
    except Exception as exc:
        print(f"  扫描失败：{exc}")
        return None


def print_scan(scan: Dict[str, Any]) -> None:
    nodes = scan.get("nodes", {})
    models = scan.get("models", [])
    mapping = scan.get("discovered_mapping", {})
    workflows = scan.get("workflows", [])
    print(f"\n  发现 {len(nodes)} 个 MiniMax 相关节点：")
    for name in sorted(nodes):
        spec = nodes[name]
        print(f"    - {name}  必填输入: {spec.get('required')}  可选输入: {spec.get('optional')}  输出: {spec.get('output')}")
    print(f"\n  自动识别的节点映射：")
    for role, cls in mapping.items():
        if cls:
            print(f"    {role} -> {cls}")
    print(f"\n  发现 {len(models)} 个模型候选：")
    for m in models[:20]:
        print(f"    - {m['model']}  (来自 {m['node']}.{m['input']})")
    if workflows:
        print(f"\n  发现 {len(workflows)} 个工作流文件：")
        for w in workflows:
            print(f"    - {w['name']}  ({w['format']} 格式, {w['node_count']} 节点, "
                  f"MiniMax 节点: {w.get('mini_nodes') or '无'})")


def choose_workflow(scan: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """从扫描结果/工作流目录中选择一个工作流接入。

    优先使用扫描结果中的工作流列表；若扫描失败则直接扫描本地工作流目录。
    """
    workflows = (scan or {}).get("workflows") or []
    if not workflows:
        dirs = [resolve_path(cfg.get("comfyui", {}).get("workflow_dir", "workflow_templates"))]
        workflows = scan_workflow_files(dirs)

    # 只展示包含 MiniMax 节点的真实工作流；内置角色占位模板不在此列出
    workflows = [w for w in workflows if w.get("mini_nodes")]
    if not workflows:
        print("\n未发现可接入的工作流 JSON，将使用内置角色占位模板 "
              "（workflow_templates/minimax_h3_workflow.json）。")
        return

    print("\n请选择要接入的视频生成工作流：")
    for i, w in enumerate(workflows, 1):
        print(f"  {i}. {w['name']}  ({w['format']} 格式, {w['node_count']} 节点, "
              f"MiniMax 节点: {w.get('mini_nodes') or '无'})")
    print(f"  {len(workflows) + 1}. 不使用，保持内置角色占位模板")

    while True:
        try:
            raw = input(f"  输入序号（回车=第 1 项）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(1)
        if not raw:
            idx = 0
        else:
            try:
                idx = int(raw) - 1
            except ValueError:
                idx = -1
        if 0 <= idx < len(workflows):
            chosen = workflows[idx]
            break
        if idx == len(workflows):
            print("  已选择内置角色占位模板。")
            return
        print("  序号无效，请重新输入。")

    src = Path(chosen["path"])
    # 复制到 exe 目录/项目目录下的 workflow_templates（可写），保证 config.yaml 相对路径可移植
    templates_dir = PROJECT_ROOT / "workflow_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    dest_name = src.name
    dest = templates_dir / dest_name
    if src.resolve() != dest.resolve():
        if dest.exists():
            dest_name = f"selected_{src.name}"
            dest = templates_dir / dest_name
        shutil.copy(src, dest)
    rel = str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")
    cfg.setdefault("comfyui", {})["workflow_template"] = rel
    cfg.setdefault("minimax_h3", {}).setdefault("comfyui", {})["workflow_template"] = rel
    print(f"  已选择工作流：{rel}")


def configure_comfyui(cfg: Dict[str, Any], provider: str) -> None:
    mh3 = cfg.setdefault("minimax_h3", {})
    mh3["provider"] = provider
    mh3.setdefault("comfyui", {})
    default_url = "http://127.0.0.1:8188" if provider == "local_comfyui" else ""
    base_url = ask("ComfyUI 地址", default=default_url, allow_empty=provider == "remote_comfyui")
    if not base_url:
        base_url = "http://127.0.0.1:8188"
    mh3["comfyui"]["base_url"] = base_url

    scan = scan_comfyui(base_url)
    model_name = ""
    node_mapping: Dict[str, str] = {}

    if scan:
        print_scan(scan)
        models = [m["model"] for m in scan.get("models", [])]
        if models:
            idx = choose(models, "请选择 MiniMax 模型（0=自动选择第一个）")
            if idx is not None and idx >= 0:
                model_name = models[idx]
        mapping = scan.get("discovered_mapping", {})
        if mapping and ask_yes_no("是否接受自动发现的节点映射？", default=True):
            node_mapping = {k: v for k, v in mapping.items() if v}
    else:
        print("  未能自动扫描。可稍后在 config.yaml 中手动配置 comfyui.node_mapping。")

    if not model_name:
        model_name = ask("MiniMax 模型名称（留空=自动选择）", default="", allow_empty=True)
    mh3["model_name"] = model_name

    if not node_mapping:
        print("\n请手动输入关键节点映射（留空=保持自动发现/自动适配）：")
        for role in ("TEXT_TO_VIDEO", "IMAGE_TO_VIDEO", "MODEL_LOADER", "SAVE_VIDEO"):
            value = ask(f"  {role} 节点 class_type", default="", allow_empty=True)
            if value:
                node_mapping[role] = value
    mh3["comfyui"]["node_mapping"] = node_mapping

    # 选择工作流接入（扫描工作流目录，可直接使用用户导出的工作流 JSON）
    choose_workflow(scan, cfg)

    # 同步到顶层兼容段
    cfg.setdefault("comfyui", {})["base_url"] = base_url
    cfg["comfyui"]["node_mapping"] = node_mapping
    cfg.setdefault("minimax", {})["model_name"] = model_name


def configure_remote_api(cfg: Dict[str, Any]) -> None:
    mh3 = cfg.setdefault("minimax_h3", {})
    mh3["provider"] = "remote_api"
    api = mh3.setdefault("api", {})
    api["base_url"] = ask("API base_url", default="https://api.minimaxi.com/v1")
    api["api_key"] = ask("API key", default="", allow_empty=True)
    api["model"] = ask("模型名称", default="MiniMax-H3")
    api["create_path"] = ask("生成任务路径 create_path", default="/video_generation")
    api["query_path"] = ask("查询任务路径 query_path", default="/query/video_generation")
    cfg.setdefault("minimax", {})["model_name"] = api["model"]


def configure_ffmpeg(cfg: Dict[str, Any]) -> None:
    found = detect_ffmpeg(cfg)
    print(f"\nffmpeg 检测结果：{'未找到' if found == 'ffmpeg' else found}")
    if found == "ffmpeg":
        path = ask("请手动输入 ffmpeg 可执行文件完整路径（留空=稍后自行安装）",
                   default="", allow_empty=True)
        cfg.setdefault("ffmpeg", {})["path"] = path
    else:
        cfg.setdefault("ffmpeg", {})["path"] = ""


def write_config(cfg: Dict[str, Any]) -> None:
    if CONFIG_PATH.exists():
        if not ask_yes_no(f"{CONFIG_PATH.name} 已存在，是否覆盖？", default=True):
            print("  已取消写入。")
            return
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"\n配置已写入：{CONFIG_PATH}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=" * 64)
    print("小说转视频 - 首次运行配置向导")
    print("=" * 64)

    # 以默认模板为底，若已有配置则在其基础上更新
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            existing = load_config(CONFIG_PATH)
            cfg = _merge(cfg, existing)
            print(f"检测到已有配置：{CONFIG_PATH}，将在此基础上更新。")
        except Exception as exc:
            print(f"读取已有配置失败（{exc}），将使用默认模板。")

    print("\n请选择 MiniMax H3 的来源模式：")
    print("  1. local_comfyui   - 本地 ComfyUI（默认）")
    print("  2. remote_comfyui  - 远程 ComfyUI 服务器")
    print("  3. remote_api      - MiniMax 官方/第三方 HTTP API")
    choice = ask("请输入序号", default="1")
    if choice == "2":
        configure_comfyui(cfg, "remote_comfyui")
    elif choice == "3":
        configure_remote_api(cfg)
    else:
        configure_comfyui(cfg, "local_comfyui")

    configure_ffmpeg(cfg)

    print("\n分镜与生成参数使用默认值（可在 config.yaml 中修改）：")
    print(f"  storyboard.mode = {cfg.get('storyboard', {}).get('mode')}")
    print(f"  generation.concurrent_jobs = {cfg.get('generation', {}).get('concurrent_jobs')}")

    write_config(cfg)
    print("\n下一步：")
    print("  python main.py --scan            # 扫描 ComfyUI 节点与模型")
    print("  python main.py --storyboard-only # 生成分镜")
    print("  python main.py --auto-run        # 一键生成视频")
    print("  python gui.py                    # 打开图形界面")
    return 0


if __name__ == "__main__":
    sys.exit(main())
