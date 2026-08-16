"""小说转视频主入口。

用法示例：
    python main.py --novel examples/sample_novel.txt --mode auto
    python main.py --novel examples/sample_novel.txt --segment-duration 10 --mode fixed_duration
    python main.py --novel examples/sample_novel.txt --auto-run
    python main.py --novel examples/sample_novel.txt --auto-run --concurrent-jobs 2 --max-retries 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.storyboard_generator import StoryboardGenerator
from modules.continuity_tracker import ContinuityTracker
from modules.comfyui_client import ComfyUIClient, ComfyUIError
from modules.pipeline_runner import PipelineRunner
from modules.video_assembler import VideoAssembler
from modules.config_utils import (load_config as _load_config,
                                  normalize_config, detect_ffmpeg,
                                  PROJECT_ROOT, RESOURCE_ROOT)

log = logging.getLogger("novel2video")


# ----------------------------------------------------------------------
# 配置与工具
# ----------------------------------------------------------------------
def load_config(path: str | Path) -> Dict[str, Any]:
    """加载并归一化 YAML 配置（兼容 minimax_h3 三种 provider 模式）。"""
    return normalize_config(_load_config(path))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def setup_logging(level: str, log_dir: str | Path) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"novel2video_{datetime.now():%Y%m%d_%H%M%S}.log"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"警告：无法创建日志文件 {log_file}：{exc}")
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("已写入 %s", path)


def collect_shot_files(storyboard: Dict[str, Any], shots_dir: Path) -> List[str]:
    files: List[str] = []
    for shot in storyboard.get("shots", []) or []:
        shot_id = shot.get("shot_id", "")
        matches = sorted(shots_dir.glob(f"{shot_id}.*"))
        video = next((str(m) for m in matches
                      if m.suffix.lower() in (".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv")),
                     None)
        if video:
            files.append(video)
        elif matches:
            files.append(str(matches[0]))
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="小说转视频：ComfyUI + MiniMax H3 自动化流水线")
    parser.add_argument("--novel", default="examples/sample_novel.txt",
                        help="小说文本文件（txt/markdown）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=["auto", "fixed_duration"],
                        help="分段策略：auto=LLM 智能切分；fixed_duration=按固定时长切分")
    parser.add_argument("--segment-duration", type=float, default=None,
                        help="fixed_duration 模式下每镜头时长（秒），如 --segment-duration 10")
    parser.add_argument("--max-shot-duration", type=float, default=None,
                        help="auto 模式下单镜头最大时长（秒）")
    parser.add_argument("--auto-run", action="store_true",
                        help="分镜完成后立即开始按顺序生成所有镜头并拼接")
    parser.add_argument("--concurrent-jobs", type=int, default=None,
                        help="并发提交 ComfyUI 的镜头数（默认 1，顺序执行保证空间衔接）")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="每个镜头最大重试次数")
    parser.add_argument("--no-reference-image", action="store_true",
                        help="禁用首帧参考图（纯文本生成）")
    parser.add_argument("--generate-first-frame", action="store_true",
                        help="通过 ComfyUI 图像工作流生成首镜头首帧")
    parser.add_argument("--skip-storyboard", action="store_true",
                        help="跳过 LLM 分镜，直接使用已有 storyboard.json")
    parser.add_argument("--storyboard-only", action="store_true",
                        help="只生成分镜 JSON 与连续性报告，不调用 ComfyUI")
    parser.add_argument("--scan", action="store_true",
                        help="扫描 ComfyUI 中的 MiniMax 节点与模型，写入缓存并打印结果")
    parser.add_argument("--wizard", action="store_true",
                        help="运行首次运行配置向导")
    parser.add_argument("--visual-qa", action="store_true",
                        help="对已有 shots/ 与 storyboard.json 执行画面层视觉质检并写入报告")
    parser.add_argument("--log-level", default=None, help="日志级别：DEBUG/INFO/WARNING/ERROR")
    return parser


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    if args.wizard:
        import wizard
        return wizard.main()

    # 1) 配置
    config = load_config(args.config)
    cli_overrides: Dict[str, Any] = {}
    if args.mode:
        cli_overrides.setdefault("storyboard", {})["mode"] = args.mode
    if args.segment_duration is not None:
        cli_overrides.setdefault("storyboard", {})["segment_duration_seconds"] = args.segment_duration
    if args.max_shot_duration is not None:
        cli_overrides.setdefault("minimax", {})["max_shot_duration"] = args.max_shot_duration
    if args.auto_run:
        cli_overrides.setdefault("generation", {})["auto_run"] = True
    if args.concurrent_jobs is not None:
        cli_overrides.setdefault("generation", {})["concurrent_jobs"] = args.concurrent_jobs
    if args.max_retries is not None:
        cli_overrides.setdefault("generation", {})["max_retries"] = args.max_retries
    if args.no_reference_image:
        cli_overrides.setdefault("storyboard", {})["use_reference_image"] = False
    if args.generate_first_frame:
        cli_overrides.setdefault("storyboard", {})["generate_first_frame"] = True
    config = deep_merge(config, cli_overrides)
    config = normalize_config(config)

    log_level = args.log_level or config.get("logging", {}).get("level", "INFO")
    setup_logging(log_level, PROJECT_ROOT / config.get("logging", {}).get("log_dir", "logs"))

    log.info("=" * 70)
    log.info("小说转视频流水线启动")
    log.info("项目目录：%s", PROJECT_ROOT)
    log.info("MiniMax H3 来源：%s", config.get("_provider", "local_comfyui"))
    log.info("模式：%s | 自动运行：%s | 并发：%s | 重试：%s",
             config.get("storyboard", {}).get("mode"),
             config.get("generation", {}).get("auto_run"),
             config.get("generation", {}).get("concurrent_jobs"),
             config.get("generation", {}).get("max_retries"))

    # --scan：扫描 ComfyUI 中的 MiniMax 节点与模型并缓存
    if args.scan:
        if config.get("_provider") == "remote_api":
            log.error("当前 provider 为 remote_api，--scan 仅适用于 local_comfyui / remote_comfyui。")
            return 1
        client = ComfyUIClient(config)
        scan = client.scan_environment()
        print(json.dumps(scan, ensure_ascii=False, indent=2))
        log.info("扫描完成：发现 %d 个 MiniMax 节点，%d 个模型候选。",
                 len(scan.get("nodes", {})), len(scan.get("models", [])))
        return 0

    # --visual-qa：对已有 shots/ 与 storyboard.json 执行画面层质检
    if args.visual_qa:
        storyboard_path = PROJECT_ROOT / "storyboard.json"
        if not storyboard_path.exists():
            log.error("storyboard.json 不存在，无法执行视觉质检。请先生成镜头。")
            return 1
        with open(storyboard_path, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
        shots_dir = PROJECT_ROOT / config.get("generation", {}).get("shots_dir", "shots")
        from modules.visual_qa import VisualQA
        vqa = VisualQA(config, storyboard, shots_dir)
        result = vqa.run()
        report = {}
        report_path = PROJECT_ROOT / "continuity_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        report["visual_qa"] = result
        write_json(report_path, report)
        log.info("视觉质检完成：检查 %d 组，漂移告警 %d，镜像告警 %d",
                 result["summary"]["checked_transitions"],
                 result["summary"]["drift_warnings"],
                 result["summary"]["mirror_warnings"])
        return 0

    # 2) 读取小说
    novel_path = Path(args.novel)
    if not novel_path.is_absolute():
        candidates = [PROJECT_ROOT / novel_path]
        if RESOURCE_ROOT != PROJECT_ROOT:
            candidates.append(RESOURCE_ROOT / novel_path)
        novel_path = next((c for c in candidates if c.exists()), candidates[0])
    if not novel_path.exists():
        log.error("小说文件不存在：%s", args.novel)
        return 1
    novel_text = novel_path.read_text(encoding="utf-8-sig")
    log.info("小说文件：%s（%d 字符）", novel_path, len(novel_text))

    shots_dir = PROJECT_ROOT / config.get("generation", {}).get("shots_dir", "shots")
    output_dir = PROJECT_ROOT / config.get("generation", {}).get("output_dir", "output")
    shots_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) 分镜
    storyboard_path = PROJECT_ROOT / "storyboard.json"
    report_path = PROJECT_ROOT / "continuity_report.json"
    if not args.skip_storyboard:
        log.info("开始自动分镜…")
        generator = StoryboardGenerator(config)
        storyboard = generator.generate(
            novel_text,
            mode=config.get("storyboard", {}).get("mode", "auto"),
            segment_duration=float(config.get("storyboard", {}).get("segment_duration_seconds", 10)),
        )
        write_json(storyboard_path, storyboard)
        report = generator.last_report or {}
        report["generated_at"] = datetime.now().isoformat(timespec="seconds")
        report["stage"] = "storyboard"
        write_json(report_path, report)
        log.info("分镜 JSON 与连续性报告已生成。")
    else:
        log.info("跳过 LLM 分镜，加载已有 storyboard.json")
        with open(storyboard_path, "r", encoding="utf-8") as f:
            storyboard = json.load(f)

    if args.storyboard_only:
        log.info("--storyboard-only 模式：流程结束。")
        return 0

    # 4) 一键顺序自动生成
    if not config.get("generation", {}).get("auto_run", False):
        log.info("未开启 --auto-run。分镜已就绪；如需自动生成视频，请运行：")
        log.info("    python main.py --auto-run")
        return 0

    provider = config.get("_provider", "local_comfyui")
    if provider == "remote_api":
        from modules.remote_api_client import RemoteMiniMaxAPIClient
        log.info("使用 MiniMax 远程 API：%s", config.get("minimax_h3", {}).get("api", {}).get("base_url"))
        client = RemoteMiniMaxAPIClient(config)
    else:
        log.info("连接 ComfyUI：%s", config.get("comfyui", {}).get("base_url"))
        try:
            client = ComfyUIClient(config)
            client.build_node_mapping()
        except ComfyUIError as exc:
            log.error(str(exc))
            log.error("请确认 ComfyUI 已启动、MiniMax H3 节点已安装，或检查 config.yaml 的 comfyui 配置。")
            return 1

    tracker = ContinuityTracker()
    tracker.initialize_from_storyboard(storyboard)

    runner = PipelineRunner(config, storyboard, client, tracker)
    try:
        results = runner.run()
    except Exception as exc:
        log.exception("生成调度器异常退出：%s", exc)
        results = runner.results

    # 5) 拼接
    final_path = output_dir / "final.mp4"
    shot_files = collect_shot_files(storyboard, shots_dir)
    if shot_files:
        log.info("开始拼接 %d 个镜头…", len(shot_files))
        assembler = VideoAssembler(config)
        try:
            assembler.assemble(shot_files, storyboard, final_path)
            log.info("最终视频已生成：%s", final_path)
        except Exception as exc:
            log.error("视频拼接失败：%s", exc)
    else:
        log.warning("没有可拼接的镜头文件。")

    # 5.5) 画面层视觉质检（默认开启）
    visual_qa_result = None
    if config.get("visual_qa", {}).get("enabled", True) and shot_files:
        from modules.visual_qa import VisualQA
        try:
            vqa = VisualQA(config, storyboard, shots_dir)
            visual_qa_result = vqa.run()
            log.info("视觉质检：检查 %d 组，漂移告警 %d，镜像告警 %d",
                     visual_qa_result["summary"]["checked_transitions"],
                     visual_qa_result["summary"]["drift_warnings"],
                     visual_qa_result["summary"]["mirror_warnings"])
        except Exception as exc:
            log.warning("视觉质检失败（不影响成片）：%s", exc)

    # 6) 连续性报告
    report = tracker.report
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["stage"] = "generation"
    report["generation_results"] = results
    if visual_qa_result is not None:
        report["visual_qa"] = visual_qa_result
    report["summary"] = {
        "shots_total": len(storyboard.get("shots", []) or []),
        "shots_success": sum(1 for r in results if r.get("status") == "success"),
        "shots_failed": sum(1 for r in results if r.get("status") == "failed"),
        "shots_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "conflicts": len(tracker.report.get("conflicts", [])),
        "corrections": len(tracker.report.get("corrections", [])),
        "reverse_shots": len(tracker.report.get("reverse_shots", [])),
        "final_video": str(final_path) if final_path.exists() else None,
    }
    if visual_qa_result is not None:
        report["summary"]["visual_drift_warnings"] = visual_qa_result["summary"]["drift_warnings"]
        report["summary"]["visual_mirror_warnings"] = visual_qa_result["summary"]["mirror_warnings"]
    write_json(report_path, report)

    failed = report["summary"]["shots_failed"]
    log.info("=" * 70)
    log.info("流水线结束。成功 %d / 失败 %d / 跳过 %d。",
             report["summary"]["shots_success"], failed, report["summary"]["shots_skipped"])
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
