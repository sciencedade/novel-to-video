# 小说转视频（ComfyUI + MiniMax H3）

将小说文本自动转换为**连续、空间一致、方向正确**的视频。系统内置“场景空间连续性状态机”，
不依赖首尾帧图像保持画面连续性——空间方位由 `spatial_anchors` + 文本提示词显式锁定。

## 特性

- **自动分镜**：调用 OpenAI 兼容 LLM 解析小说，切分场景/镜头；支持 `auto`（智能切分）与
  `fixed_duration`（按 `--segment-duration 10` 固定时长切分）两种模式
- **空间连续性状态机**：维护场景级固定设施布局（门/窗/家具 左/中/右）与角色级状态
  （位置、朝向、运动方向），逐镜头校验并自动修正冲突
- **方向锁 / 180 度规则 / 反打镜头标注 / 镜像检测**
- **首尾帧衔接**：顺序模式下抽取上一镜头真实尾帧作为下一镜头首帧参考图，
  并同时在 `video_prompt` 中显式写入空间锁定文本（图像参考只作为风格约束）
- **ComfyUI MiniMax H3 调用器**：启动时通过 `/object_info` 动态发现节点，
  不硬编码不存在的节点名；支持可配置节点映射
- **一键顺序自动生成**：断点续传、失败自动重试（重试时注入空间锚点修正）、
  可选并发、进度显示、自动 ffmpeg 拼接
- **桌面 GUI**：`python gui.py` 提供参数配置、一键分镜/一键生成、实时日志、进度条与停止按钮

## 目录结构

```
novel2video/
├── main.py                      # 主入口（命令行）
├── gui.py                       # 桌面 GUI（tkinter）
├── wizard.py                    # 首次运行配置向导
├── install.bat / setup.sh       # 一键安装并启动向导
├── build_exe.bat                # 一键打包 Windows exe（PyInstaller）
├── requirements.txt / requirements-build.txt
├── config.yaml                  # 配置模板（无硬编码绝对路径）
├── modules/
│   ├── config_utils.py          # 配置归一化 / ffmpeg 检测 / 路径解析
│   ├── workflow_scanner.py      # 工作流扫描 / UI→API 转换 / 角色识别
│   ├── storyboard_generator.py  # 自动分镜
│   ├── continuity_tracker.py    # 空间连续性状态机（核心）
│   ├── comfyui_client.py        # ComfyUI 调用器（节点/模型/工作流扫描）
│   ├── remote_api_client.py     # MiniMax 远程 API 适配器
│   ├── pipeline_runner.py       # 一键顺序自动生成调度器
│   └── video_assembler.py       # ffmpeg 拼接
├── workflow_templates/
│   ├── minimax_h3_workflow.json # MiniMax H3 工作流模板（角色占位符）
│   └── first_frame_workflow.json# 可选首帧图像生成工作流
└── examples/sample_novel.txt    # 示例小说章节
```

## 安装

### 方式一：一键安装并启动配置向导

Windows 双击或运行：

```bat
install.bat
```

Linux / macOS：

```bash
chmod +x setup.sh
./setup.sh
```

脚本会创建虚拟环境、安装依赖并自动运行 `python wizard.py` 配置向导。

### 方式二：手动安装

```bash
cd novel2video
python -m venv venv
# Windows: venv\Scripts\activate     Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python wizard.py      # 首次运行配置向导
# 确认 ffmpeg 在 PATH 中；检测不到时向导会让你手动指定 ffmpeg.path
```

### 方式三：打包为 Windows exe（推荐分发）

```bat
cd novel2video
build_exe.bat
```

打包完成后在 `dist\` 生成：

| 文件 | 说明 |
|------|------|
| `Novel2Video.exe` | 图形界面（无控制台窗口），日常使用 |
| `Novel2Video-CLI.exe` | 命令行/配置向导（`--wizard`、`--scan`、`--auto-run`） |

```bat
Novel2Video-CLI.exe --wizard     # 首次运行配置向导
Novel2Video-CLI.exe --scan       # 扫描 ComfyUI 节点/模型/工作流
Novel2Video-CLI.exe --auto-run   # 命令行一键生成
```

打包机制：

- `workflow_templates/`、`examples/`、`config.yaml` 作为只读资源打进 exe
- 运行后 `config.yaml`、`output/`、`shots/`、`logs/`、`cache/` 均生成在 exe 所在目录
- 若 exe 目录没有 `config.yaml`，自动使用内置模板（也可点 GUI 的“配置向导”生成）
- 工作流模板优先读取 exe 目录下 `workflow_templates/`，其次读取打包资源

> 注意：`ffmpeg` 不会被自动打包进 exe（GPL 许可证原因），请用户自行安装并加入 PATH，
> 或在配置向导中手动指定 `ffmpeg.path`。

## 三种模型来源模式

`config.yaml` 的 `minimax_h3.provider` 决定 MiniMax H3 从哪里生成视频：

### 1. local_comfyui（本地 ComfyUI，默认）

适合在本机部署了 ComfyUI + MiniMax H3 节点（如 `MinimaxTextToVideoNode`、
`MinimaxImageToVideoNode`、`MiniMaxH3ImageToVideo` 等）的用户。

```yaml
minimax_h3:
  provider: "local_comfyui"
  comfyui:
    base_url: "http://127.0.0.1:8188"
    node_mapping:
      TEXT_TO_VIDEO: ""      # 留空 = 自动发现
      IMAGE_TO_VIDEO: ""
  model_name: ""             # 留空 = 自动取节点第一个模型
```

启动时程序调用 `GET /object_info` 扫描节点与模型，结果缓存到
`cache/comfyui_scan_cache.json`。也可手动执行 `python main.py --scan` 查看扫描结果。

### 2. remote_comfyui（远程 ComfyUI）

ComfyUI 部署在服务器/NAS/云主机上，本机通过 HTTP 访问：

```yaml
minimax_h3:
  provider: "remote_comfyui"
  comfyui:
    base_url: "http://192.168.1.50:8188"   # 或 https://your-server:8188
    node_mapping:
      TEXT_TO_VIDEO: "MinimaxTextToVideoNode"   # 建议显式填写，避免自动发现偏差
      IMAGE_TO_VIDEO: "MinimaxImageToVideoNode"
  model_name: "T2V-01"
```

注意：远程 ComfyUI 必须开放端口且允许跨域/防火墙访问；输出视频通过 `/view` 下载。

### 3. remote_api（MiniMax 官方/第三方 API）

不依赖 ComfyUI，直接调用 MiniMax 官方或兼容的视频生成 HTTP API：

```yaml
minimax_h3:
  provider: "remote_api"
  api:
    base_url: "https://api.minimaxi.com/v1"
    api_key: "your-api-key"
    model: "MiniMax-H3"
    create_path: "/video_generation"
    query_path: "/query/video_generation"
    timeout_seconds: 600
    poll_interval_seconds: 3
```

默认按 MiniMax 官方风格实现（`POST /video_generation` 提交，`GET /query/video_generation`
轮询）。第三方兼容 API 只要支持同样风格的提交/查询与 `task_id` 返回，即可通过
`base_url`、`create_path`、`query_path` 适配。

> 注意：`remote_api` 模式不生成首帧参考图（`generate_first_frame` 无效），
> 空间连续性完全由 `video_prompt` 中的文本锚点锁定。

## 首次运行配置向导

```bash
python wizard.py
```

向导会：

1. 询问 MiniMax H3 来源：本地 ComfyUI / 远程 ComfyUI / 远程 API
2. ComfyUI 模式下自动扫描 `/object_info`，列出 MiniMax 节点与模型供选择
3. **扫描工作流目录，列出包含 MiniMax 节点的工作流 JSON，让你直接选择接入**
   （支持 ComfyUI 的 API 格式与普通 Save 的 UI 格式，UI 格式自动转换）
4. 远程 API 模式下要求输入 `base_url`、`api_key`、`model`
5. 自动检测 `ffmpeg`；检测不到时要求手动指定路径
6. 生成/更新 `config.yaml`

## 工作流扫描与接入

项目内置了角色占位模板 `workflow_templates/minimax_h3_workflow.json` 作为兜底。
但更推荐**使用你自己导出的 MiniMax 工作流**：

1. 在 ComfyUI 中搭好 MiniMax 视频生成工作流（文生视频/图生视频 + 保存节点）
2. 用 “Save (API Format)” 或普通 “Save” 导出 JSON
3. 把 JSON 放进 `workflow_templates/`（或运行向导，选择该文件后自动复制进来）
4. 运行 `python wizard.py`，向导会扫描并列出这些工作流供你选择
5. 或手动在 `config.yaml` 中指定：

```yaml
comfyui:
  workflow_dir: "workflow_templates"                 # 扫描目录
  workflow_template: "workflow_templates/my_minimax.json"  # 选定工作流
```

运行时 `ComfyUIClient` 会：

- 自动识别工作流中的 MiniMax 节点角色（TEXT_TO_VIDEO / IMAGE_TO_VIDEO /
  MODEL_LOADER / SAVE_VIDEO），**不替换节点 class_type**
- 只把 `prompt / seed / start_image / end_image / filename_prefix` 等参数
  注入识别到的节点，保留工作流里你已选好的模型与其它连接
- 若工作流是 UI 格式，先转换为 API 格式再接入

`python main.py --scan` 的扫描结果（`cache/comfyui_scan_cache.json`）中也包含
`workflows` 字段，列出每个工作流的格式、节点数与识别到的 MiniMax 节点。

## 快速开始

```bash
# 0. 扫描 ComfyUI 中的 MiniMax 节点与模型（写入 cache/comfyui_scan_cache.json）
python main.py --scan

# 1. 只生成分镜（不连接 ComfyUI，未配置 LLM 时自动使用本地确定性分镜）
python main.py --novel examples/sample_novel.txt --mode auto

# 2. 固定 10 秒一段
python main.py --novel examples/sample_novel.txt --mode fixed_duration --segment-duration 10

# 3. 一键全自动：分镜 -> 逐个生成 -> 拼接 final.mp4
python main.py --novel examples/sample_novel.txt --auto-run

# 4. 断点续传 + 重试 + 并发
python main.py --novel examples/sample_novel.txt --auto-run --max-retries 5 --concurrent-jobs 1

# 5. 使用已有 storyboard.json 继续生成
python main.py --skip-storyboard --auto-run

# 6. 启动桌面 GUI（参数配置、实时日志、进度条、停止按钮）
python gui.py
```

## 输出

| 文件 | 说明 |
|------|------|
| `storyboard.json` | 完整分镜表（每个镜头含 `spatial_anchors`、`continuity_input/output`、`video_prompt`） |
| `shots/shot_001.mp4` ... | 每个镜头视频 |
| `shots/frames/` | 顺序模式抽取的上一镜头尾帧（首帧参考图） |
| `output/final.mp4` | 拼接后的完整视频 |
| `continuity_report.json` | 空间连续性校验报告（冲突/修正/反打镜头记录） |
| `logs/novel2video_*.log` | 运行日志 |

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--novel` | 小说 txt/markdown 路径 |
| `--config` | 配置文件路径（默认 `config.yaml`） |
| `--mode auto/fixed_duration` | 分段策略 |
| `--segment-duration 10` | `fixed_duration` 模式每镜头秒数 |
| `--max-shot-duration 10` | `auto` 模式单镜头最大秒数 |
| `--auto-run` | 分镜后立即自动生成所有镜头并拼接 |
| `--concurrent-jobs N` | 并发提交数（默认 1，顺序执行保证空间衔接） |
| `--max-retries N` | 每镜头最大重试次数 |
| `--no-reference-image` | 禁用首帧参考图，纯文本生成 |
| `--generate-first-frame` | 通过 ComfyUI 图像工作流生成首镜头首帧 |
| `--skip-storyboard` | 跳过 LLM 分镜，使用已有 `storyboard.json` |
| `--storyboard-only` | 只生成分镜与连续性报告 |
| `--scan` | 扫描 ComfyUI 的 MiniMax 节点与模型，写入缓存并打印 |
| `--log-level` | DEBUG/INFO/WARNING/ERROR |

## 节点映射配置

工作流模板中的节点使用**角色占位符**（`__MODEL_LOADER__`、`__TEXT_TO_VIDEO__` 等），
运行时由 `comfyui_client` 调用 ComfyUI `/object_info` 自动发现实际节点名并替换。
无需修改模板即可适配不同 MiniMax H3 节点包。

自动发现规则（按节点 class_type 小写匹配）：

| 角色 | 匹配规则 |
|------|----------|
| `MODEL_LOADER` | MiniMax/H3 节点中同时含 `loader` 或 `model` |
| `TEXT_TO_VIDEO` | MiniMax/H3 节点中同时含 `text` + `video`（或 `t2v`） |
| `IMAGE_TO_VIDEO` | MiniMax/H3 节点中同时含 `image` + `video`（或 `i2v`） |
| `SAVE_VIDEO` | 含 `save`+`video`，或 `VHS_VideoCombine` 等 |
| `TEXT_TO_IMAGE` / `SAVE_IMAGE` | 首帧图像生成工作流使用（可选） |

若自动发现结果不符合你的节点包，在 `config.yaml` 中显式覆盖（推荐写入
`minimax_h3.comfyui.node_mapping`，与顶层 `comfyui.node_mapping` 等价且优先级更高）：

```yaml
minimax_h3:
  provider: "local_comfyui"
  comfyui:
    base_url: "http://127.0.0.1:8188"
    node_mapping:
      MODEL_LOADER: "MiniMaxH3ModelLoader"
      TEXT_TO_VIDEO: "MiniMaxH3TextToVideo"
      IMAGE_TO_VIDEO: "MiniMaxH3ImageToVideo"
      SAVE_VIDEO: "VHS_VideoCombine"
```

`comfyui.input_name_overrides` 可覆盖模板参数到节点实际输入名的映射（一般无需设置；
客户端会自动把 `prompt/text/positive_prompt`、`width/height`、`fps`、`num_frames` 等
常见输入名匹配到目标节点）。

## 空间连续性硬性规则

1. 同一场景中，固定设施（门、窗、楼梯、主要家具）的屏幕方位在连续镜头中必须一致。
2. 角色运动方向在连续动作中必须一致；上一镜头从左向右走，下一镜头不能突然从右向左，
   除非剧情需要并在 `action` 中显式说明（如“掉头”）。
3. 反打镜头必须遵循 180 度规则，并在提示词中显式标注：
   `反打镜头：摄影机已移到对面，门现在在画面右侧……`
4. 禁止使用镜像画面衔接；镜像会被检测并自动修正。
5. 每个镜头生成前，`ContinuityTracker.prepare_shot()` 检查 `spatial_anchors` 与
   `scene_layout` 是否冲突；冲突自动修正提示词并记录到 `continuity_report.json`。

## LLM 配置

使用任意 OpenAI 兼容接口：

```yaml
llm:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
```

未配置 `api_key` 时自动回退为本地确定性分镜（按 `chars_per_second` 估算时长切分文本），
保证离线也能产出完整 `storyboard.json`。

## MiniMax H3 工作流模板

模板 `workflow_templates/minimax_h3_workflow.json` 使用 ComfyUI API 格式，节点为角色占位符。
提交前客户端会：

1. 加载模板
2. 通过 `/object_info` 校验目标节点的实际输入名
3. 将 `prompt / width / height / fps / num_frames / seed / start_image / end_image` 等
   值注入匹配的输入；不存在的输入自动移除，避免无效参数报错
4. 自动选择 `minimax.model_name`（若留空则取模型加载节点的第一个可选模型）

## 并发说明

- `concurrent_jobs = 1`（默认）：**严格顺序**。上一镜头完成后抽取真实尾帧作为下一镜头
  首帧参考图，空间连续性最强。
- `concurrent_jobs > 1`：提示词仍按顺序由状态机生成，多个任务并发提交到 ComfyUI；
  参考图链路退化为 `storyboard` 中静态 `reference_image`（因为下一镜头提示词生成时
  上一镜头可能尚未完成），空间一致性由文本锚点保证。

## 常见问题

- **`无法访问 ComfyUI /object_info`**：确认 ComfyUI 已启动且 `base_url` 正确；
  若使用远程地址，确认端口开放。
- **`未发现 MODEL_LOADER 节点`**：确认已安装 MiniMax H3 节点包，或在 `config.yaml`
  的 `comfyui.node_mapping` 中手动指定节点 class_type（用 `/object_info` 查询）。
- **生成视频没有声音**：当前流水线只处理画面，旁白/对白以 `narration` 字段保存在
  `storyboard.json`，可用 TTS 单独生成后合成；开启 `ffmpeg.subtitles: true` 可先烧录字幕。
- **拼接失败**：确认 ffmpeg 在 PATH 中；不同镜头分辨率应保持 `minimax.resolution` 一致。
