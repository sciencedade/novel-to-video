# 角色定妆照资产目录

把每个角色的定妆照放在这里，然后在 `config.yaml` 的 `characters` 段引用：

```yaml
characters:
  - name: "林晚"
    reference_image: "assets/characters/林晚/front.png"
    description: "young woman, oval face, black long hair, beige coat"
    forbidden_changes: "no beard, no glasses, no hair change, no costume change"
```

推荐资产规范（行业共识）：

- 每角色至少 1 张正面/¾ 脸定妆照，理想 3–4 张（正/¾/侧/全身）
- 纯背景、均匀打光、单一签名服装
- 图片命名建议：`角色名_角度.png`，如 `林晚_front.png`、`林晚_side.png`

## 当前行为

- 分镜生成后，同角色镜头若没有显式 `reference_image`，会用该角色的定妆照作为首帧参考图
- 定妆描述与 `forbidden_changes` 会写进每个镜头的 `video_prompt`（提示词锁定）
- 后续同场景镜头仍优先使用上一镜头真实尾帧做首帧接力；场景切换/首镜回退到定妆照
