# 画面层视觉质检（Visual QA）设计文档

> 状态：**设计稿 v1（未实现）**
> 目标：把“提示词层空间连续性”补成“画面层可回读校验”，
> 让 `continuity_report.json` 里不仅有“状态机修正记录”，还有“生成后帧间漂移/镜像告警”。

---

## 1. 背景

当前空间连续性状态机是**文本层监督员**：

- 检测/修正 `spatial_anchors`、运动方向、反打镜头；
- 把修正结果注入 `video_prompt`；
- 但它**不读取生成后的画面**，无法确认 H3 是否真的把“门画在左侧”。

本设计增加一个轻量、非阻塞的**画面层质检环**：

```
storyboard → 状态机修正 → ComfyUI 生成镜头
                                    ↓
                    抽帧（首/尾帧）
                                    ↓
        相邻同场景镜头帧间相似度 / 水平镜像检测
                                    ↓
            写入 continuity_report.json → 人工/自动审阅
```

---

## 2. 目标

- 自动检测**相邻镜头间可感知的空间漂移**（门/窗/角色位置突变、构图跳变）。
- 自动检测**疑似镜像翻转**（同机位、无反打说明时，左右镜像相似度异常高）。
- 对**显式反打镜头（camera_side=B）**不误报镜像。
- 所有结果只**告警**，不阻塞生成、不自动重切镜头。
- 与状态机报告合并到同一份 `continuity_report.json`，便于统一复盘。

---

## 3. 非目标（第一版不做）

- 不做语义级检测（例如“识别出门在哪、窗在哪”）。
- 不做目标检测/物体追踪。
- 不改变生成流程，不做自动重生成。
- 不替代角色定妆照/角色 LoRA（那是一致性锚的上游方案）。

---

## 4. 运行时机

按已确认的选择：**生成后自动跑**，同时保留手动入口。

- **自动**：PipelineRunner 全部镜头生成完成后、ffmpeg 拼接前执行。
- **手动**：`python main.py --visual-qa`，对已有 `shots/` 与 `storyboard.json` 单独执行。
- GUI：生成完成后自动执行，并在日志区展示告警条目。

---

## 5. 输入 / 输出

### 输入

| 输入 | 来源 |
|---|---|
| `shots/shot_XXX.mp4`（或 webm/mov） | 生成结果 |
| `storyboard.json` | shot_id、scene_id、camera_side、spatial_anchors |
| `continuity_report.json` | 状态机报告（反打/冲突/修正记录） |
| ffmpeg | 抽帧 |

### 输出

在 `continuity_report.json` 新增 `visual_qa` 段：

```json
{
  "visual_qa": {
    "enabled": true,
    "generated_at": "2026-08-15T22:00:00",
    "transitions": [
      {
        "shot_a": "shot_001",
        "shot_b": "shot_002",
        "scene_id": "scene_01",
        "camera_side_a": "A",
        "camera_side_b": "A",
        "normal_similarity": 0.91,
        "flipped_similarity": 0.74,
        "mirror_ratio": 0.81,
        "drift_warning": false,
        "mirror_warning": false,
        "note": ""
      },
      {
        "shot_a": "shot_004",
        "shot_b": "shot_005",
        "scene_id": "scene_01",
        "camera_side_a": "B",
        "camera_side_b": "A",
        "normal_similarity": 0.42,
        "flipped_similarity": 0.86,
        "mirror_ratio": 2.05,
        "drift_warning": true,
        "mirror_warning": true,
        "note": "相邻镜头相似度低，疑似镜像翻转；若 shot_005 为未标注机位翻面，需人工确认"
      }
    ],
    "summary": {
      "checked_transitions": 8,
      "drift_warnings": 2,
      "mirror_warnings": 1
    }
  }
}
```

---

## 6. 处理流程

### 6.1 抽帧

- 每个镜头抽两帧：**首帧**、**尾帧**。
- 命令：
  ```bash
  # 首帧
  ffmpeg -y -i shot_001.mp4 -vf "select=eq(n\,0)" -frames:v 1 -q:v 2 first.jpg
  # 尾帧
  ffmpeg -y -sseof -0.5 -i shot_001.mp4 -frames:v 1 -q:v 2 last.jpg
  ```
- 抽帧结果缓存到 `shots/frames/qa/`，同名文件存在则跳过（断点友好）。

### 6.2 预处理

- 居中裁剪到正方形，再缩放到统一尺寸（默认 64×64）。
- 转灰度。
- 可选用 CLAHE 增强对比度，降低光照波动影响。

### 6.3 相似度指标

- **dHash（差异哈希）**：16×16 灰度图，逐像素差分，得到 256 bit；用汉明距离转相似度。
- **归一化互相关系数（NCC）**：像素级相关，对整体亮度变化不敏感。
- **直方图相关性**：颜色/灰度直方图，作为辅助。
- 最终 `similarity = 0.6 * dHash_sim + 0.4 * NCC`（权重可配置）。

### 6.4 镜像检测

- 对后一镜头首帧做**水平翻转**，再计算与前一镜头尾帧的相似度 `flipped_sim`。
- 同时计算不翻转的 `normal_sim`。
- 判定：
  ```
  mirror_ratio = flipped_sim / max(normal_sim, eps)
  若 mirror_ratio >= mirror_ratio_threshold（默认 1.15）
      且 camera_side_a == camera_side_b
      且 shot_b 无显式反打标注
      → mirror_warning = true
  ```

### 6.5 漂移检测

- 对**同场景、同机位**的相邻镜头（`scene_id` 相同且 `camera_side` 相同）：
  - 比较 `shot_a` 尾帧与 `shot_b` 首帧。
  - 若 `normal_sim < drift_threshold`（默认 0.85）→ `drift_warning = true`。
- 对**反打镜头对**（A→B 或 B→A）：
  - 不按普通漂移阈值告警；仅记录 `note`，因为画面本就应翻转。

### 6.6 与状态机报告合并

- `visual_qa.transitions[].camera_side_a/b` 从状态机 `continuity_output`/`shot.camera_side` 读取。
- 若某对镜头在状态机报告里已被标记为 `reverse_shot_annotated`，则镜像检测自动跳过 warning。
- 最终 `continuity_report.json` 结构：
  ```
  {
    conflicts: [...],
    corrections: [...],
    reverse_shots: [...],
    visual_qa: { ... },   // 新增
    summary: { ..., visual_drift_warnings, visual_mirror_warnings }
  }
  ```

---

## 7. 配置

在 `config.yaml` 新增段：

```yaml
visual_qa:
  enabled: true          # 生成后自动执行
  extract_frames: true
  resize: 64             # 预处理尺寸
  hash_size: 16          # dHash 块数
  drift_threshold: 0.85
  mirror_ratio_threshold: 1.15
  min_file_size_kb: 10   # 小于该值视为无效镜头，跳过
```

CLI：

```bash
python main.py --visual-qa
```

GUI：生成完成自动执行；结果写入日志区与 `continuity_report.json`。

---

## 8. 模块设计

新增文件：`modules/visual_qa.py`

```
class VisualQA:
    def __init__(self, config, storyboard, shots_dir)
    def run(self) -> dict                # 返回 visual_qa 段
    def _extract_frame(self, video, kind, out) -> Path
    def _frame_similarity(self, a, b) -> float
    def _mirror_similarity(self, a, b) -> float
    def _analyze_transition(self, a_info, b_info) -> dict
```

依赖：`ffmpeg`、`Pillow`（已存在）。不新增重依赖。

---

## 9. 测试计划

### 单元测试（无需真实视频）
- 用 PIL 生成两张相同图 → 相似度应 ≈1，无告警。
- 生成左右翻转图 → `mirror_ratio` 应明显 >1，触发 mirror_warning。
- 生成不同场景图 → `normal_sim` 低，触发 drift_warning。
- 构造显式反打 transition → 不触发 mirror_warning。

### 集成测试
- mock 生成 3 个真实短视频（同场景同机位 / 反打 / 不同场景）。
- 验证 `visual_qa` 段写入 `continuity_report.json`。

### 真实硬件验收
- 用已完成的《Last Warning》/《The Caller》镜头跑 `--visual-qa`，人工比对告警是否命中肉眼可见的漂移/镜像。

---

## 10. 已知局限

- **只能检测“变化”，不能判断“对错”**：低相似度可能是合理的运镜/转场，需要人工确认。
- **对镜头间大范围运镜/变焦敏感**：可能产生误报，阈值需按实际工作流调。
- **不读语义**：仍无法确认“门是不是真的在左侧”；语义级验证需引入 VLM（如 MiMo Vision / 多模态模型），属于后续增强。
- **依赖 ffmpeg**：无 ffmpeg 时自动跳过并记录 warning。

---

## 11. 后续增强（不在 v1）

1. 用 VLM 对首/尾帧做语义空间描述，与 `spatial_anchors` 交叉校验（如“门在左/右”）。
2. 帧序列采样（不只首尾），检测镜头内方向突变。
3. 与角色定妆照/参考图资产联动，校验角色跨镜头外观漂移。
