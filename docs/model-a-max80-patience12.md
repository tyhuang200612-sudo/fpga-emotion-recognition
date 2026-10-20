# Model A — 当前批准的训练规则（2026-10-21）

当前唯一准备训练的表情类别：`angry=0`、`happy=1`、`neutral=2`。旧 `neutral/happy/sad` 的模型权重不能直接作为 A 模型使用，也不能跨类别比较 Macro-F1。

- FPGA 目标学生结构：`48×48×1`，TinyCNN `16/32/64/64`，GAP1，三类 logits；从零训练，不加载旧 C2/C9 模型。
- 先完成 Kaggle 私有 CPU 数据审计 `tyhuanga/fpga-a-live-audit-v2`。验证原始训练数据中18%的类别分层验证集，保留官方 TEST 用于最终独立评估，不用于模型选择；经解码像素哈希过滤可证明的重复/泄漏。
- 数据审核通过后发布仅含审计元数据和 CSV 清单的私有数据集 `tyhuanga/fpga-a-angry-happy-neutral-clean-v1`。不要把原始图片提交到 GitHub。
- 两种独立随机种子 `20261021` / `20261022`，每种子**最多80 Epoch**，而不是必须满80。整个实验最多160 Epoch。
- 每轮验证 Macro-F1 对“上次显著改善记录”若超过**绝对0.001**（0.1 个百分点）则重置早停计数；连续**12轮**未达到阈值则触发 `EARLY_STOP`。最佳 `best.pt` 独立记录该种子在验证集上最高 Macro-F1，而不是最后一轮。
- ReduceLROnPlateau `patience=4,factor=0.5`，初始 `AdamW lr=0.0007`；训练轮数的上限和早停与学习率调度彼此独立。
- 私有 Kaggle Notebook 最多有**32个5轮阶段单元**，提前停下的后续单元 `STAGE_SKIPPED`，不会运行 GPU 反向传播。
- 每轮以及首批次、约每40批的 `A_TRAIN_LOG`（train loss / val accuracy / val Macro-F1 / 三类召回 / 混淆矩阵）写入 `live-progress.jsonl`，并尝试对数值-only 外部进度通道发消息。公网匿名主题不可用于隐私数据；Kaggle API可能缓冲 stdout，不能保证其 Logs 页实时更新。
- 每个实际训练阶段保存 `stage_XX.pt` 和累积 `history.json`。每种子有一个验证集最佳 `best.pt`；`results.json` 记录真实 `epochs_trained` / `stages_completed` / `stop_reason` / `best_epoch`。
- `.github/workflows/fpga-a-results.yml` 必须验证两份最佳权重、每种子实际训练轮数连续完整的历史、每个实际阶段检查点及 `EPOCH_END/STAGE_END/SEED_END/COMPLETE` 事件计数。早停可产生小于160轮的合法结果；**不要再按旧的70轮验收**。
- 完整结果**不表示手机真实摄像头/独立TEST/INT8/FPGA实现已经验收**，也不得未经测试就自动替换线上旧模型。

[实时数值日志（不含图像）](https://ntfy.sh/fpga-a-live-c16bd8934720aa5e9182de3df605940a)

[Model A GPU Notebook](https://www.kaggle.com/code/tyhuanga/fpga-model-a-angry-happy-neutral)
