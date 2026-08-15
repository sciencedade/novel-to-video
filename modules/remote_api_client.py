"""MiniMax 远程 API 客户端适配器。

实现与 ComfyUIClient 相同的调用接口（build_workflow / submit_workflow /
wait_for_completion / download_output），使 PipelineRunner 无需修改即可支持
remote_api 模式。

默认按 MiniMax 官方视频生成 API 风格实现：
    POST {base_url}/video_generation   body: {model, prompt, first_frame_image?}
    GET  {base_url}/query/video_generation?task_id=...
路径与字段均可在 config.yaml 的 minimax_h3.api 中覆盖。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)


class RemoteMiniMaxAPIClient:
    """MiniMax 官方/第三方 HTTP API 客户端（PipelineRunner 兼容接口）。"""

    def __init__(self, config: Dict[str, Any]):
        api = config.get("_remote_api") or (config.get("minimax_h3") or {}).get("api") or {}
        self.base_url = str(api.get("base_url", "https://api.minimaxi.com/v1")).rstrip("/")
        self.api_key = str(api.get("api_key", ""))
        self.model = str(api.get("model") or config.get("minimax", {}).get("model_name") or "MiniMax-H3")
        self.create_path = str(api.get("create_path", "/video_generation"))
        self.query_path = str(api.get("query_path", "/query/video_generation"))
        self.timeout = float(api.get("timeout_seconds", 600))
        self.poll_interval = float(api.get("poll_interval_seconds", 3))
        self.session = requests.Session()
        self._last_video_url: Optional[str] = None

    # ------------------------------------------------------------------
    # PipelineRunner 兼容接口
    # ------------------------------------------------------------------
    def build_node_mapping(self) -> Dict[str, Any]:
        return {}

    def ping(self) -> bool:
        return bool(self.base_url)

    def build_workflow(self, shot: Dict[str, Any], prompt: str,
                       start_image: Optional[str] = None,
                       end_image: Optional[str] = None,
                       seed: Optional[int] = None) -> Dict[str, Any]:
        """构造 API 请求体。"""
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        if seed is not None:
            payload["seed"] = seed
        if start_image:
            payload["first_frame_image"] = start_image
        if end_image:
            payload["last_frame_image"] = end_image
        return payload

    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        """提交生成任务，返回 task_id。"""
        url = f"{self.base_url}{self.create_path}"
        headers = self._headers()
        try:
            r = self.session.post(url, json=workflow, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise RuntimeError(f"远程 API 提交失败（{url}）：{exc}") from exc
        task_id = self._extract_task_id(data)
        if not task_id:
            raise RuntimeError(f"远程 API 响应缺少 task_id：{data}")
        log.info("远程 API 任务已提交：task_id=%s", task_id)
        self._last_video_url = None
        return str(task_id)

    def wait_for_completion(self, task_id: str) -> Dict[str, Any]:
        """轮询任务直到完成，返回与 ComfyUI history 类似的 outputs 结构。"""
        deadline = time.time() + self.timeout
        headers = self._headers()
        consecutive_errors = 0
        while time.time() < deadline:
            url = f"{self.base_url}{self.query_path}?task_id={task_id}"
            try:
                r = self.session.get(url, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        f"远程 API 查询连续失败 {consecutive_errors} 次（{url}）：{exc}") from exc
                if consecutive_errors == 1:
                    log.warning("轮询远程 API 失败：%s，继续等待…", exc)
                else:
                    log.debug("轮询远程 API 失败（%d 次）：%s", consecutive_errors, exc)
                time.sleep(self.poll_interval)
                continue

            inner = data.get("data", data)
            status = str(inner.get("status", "")).lower()
            if status in ("success", "succeed", "succeeded", "completed", "done", "finished"):
                video_url = self._extract_video_url(inner)
                if not video_url:
                    raise RuntimeError(f"远程 API 任务完成但响应缺少视频 URL：{data}")
                self._last_video_url = video_url
                return {"outputs": {"api": {"videos": [{"filename": video_url}]}}}
            if status in ("failed", "fail", "error", "cancelled", "canceled"):
                raise RuntimeError(f"远程 API 任务失败：{str(data)[:500]}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"等待远程 API 任务 {task_id} 完成超时（{self.timeout}s）")

    def download_output(self, entry: Dict[str, Any], dest_dir: str | Path,
                        shot_id: str) -> Optional[Path]:
        """下载远程视频到本地 shots 目录。"""
        try:
            outputs = entry.get("outputs", {})
            videos = outputs.get("api", {}).get("videos", [])
            url = videos[0].get("filename") if videos else None
        except Exception:
            url = None
        url = url or self._last_video_url
        if not url:
            log.warning("远程 API 没有可下载的视频 URL。")
            return None
        dest = Path(dest_dir) / f"{shot_id}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = self.session.get(url, timeout=600, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        log.info("远程视频已下载：%s", dest)
        return dest

    def generate_first_frame(self, prompt: str, shot_id: str,
                             dest_dir: str | Path = "shots/frames") -> Optional[str]:
        """远程 API 模式不生成首帧图像（由 API 内部处理），返回 None。"""
        return None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_task_id(data: Any) -> Optional[str]:
        if isinstance(data, dict):
            for key in ("task_id", "taskId", "id"):
                if data.get(key):
                    return str(data[key])
            inner = data.get("data")
            if isinstance(inner, dict):
                for key in ("task_id", "taskId", "id"):
                    if inner.get(key):
                        return str(inner[key])
        return None

    @staticmethod
    def _extract_video_url(data: Any) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        for key in ("video_url", "file_url", "url", "video", "video_file"):
            if data.get(key):
                return str(data[key])
        urls = data.get("video_urls") or data.get("files") or data.get("videos")
        if isinstance(urls, list) and urls:
            first = urls[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for key in ("url", "video_url", "file_url"):
                    if first.get(key):
                        return str(first[key])
        return None
