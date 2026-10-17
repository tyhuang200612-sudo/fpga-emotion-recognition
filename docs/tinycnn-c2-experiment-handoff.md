# TinyCNN-C2 同规模识别优化 · 实验协议与交接状态

## 目标与不可破坏的基准

- 解决 **neutral vs sad** 互相误判，不把情绪分类结果等同于真实情绪。
- 部署学生模型始终采用 **TinyCNNC 16/32/64/64，48×48 灰度，3 输出**；四层卷积/GAP/FC3 不变。
- 当前锁定基准：C Seed 2026 FP32 独立测试 Accuracy **74.99%**、INT8 独立测试 Accuracy **74.88%**。这两个结果来自先前完成的独立测试，不是 C2 成绩。
- 当前旧 C 权重、INT8 导出和安卓测试 APK 绝不覆盖。任何候选替换需通过验证集选择、多种子复现和完整量化对照。
- 原始独立测试集 **4,251 张**，C2 搜索阶段 **严格不加载或评估测试集**。本轮只比较原始冻结验证集 **2,979 张**。
- 当前 C2 不自动给图片贴新标签：先前 1,216 张标记只是机器提出的复核线索，必须由人审核；错误样本可以是标签噪声，也可能是模型识别能力问题。

## 三组对照实验

| 实验 | 保持不变 | 只改变的训练侧因素 |
|---|---|---|
| C2-A | TinyCNNC 学生模型结构 | 基于仅训练集的 baseline 预测实施**温和、带上限**的困难样本加权采样；不把模型预测当新标签 |
| C2-B | TinyCNNC 学生模型结构 | 类别加权交叉熵与 Focal Loss 混合，突出低置信困难样本 |
| C2-C | TinyCNNC 学生模型结构 | 训练阶段引入更宽的 Teacher，使用温度 2 的蒸馏损失；Teacher **不能**部署到 FPGA |

三组均从同一个**真实** C2026 基线 checkpoint 进行微调，保持输入预处理和类别顺序一致；教师仅训练阶段使用。每组最多 24 Epoch，patience=7；教师最多 32 Epoch，patience=8。所有数值是实验配置而非未来性能保证。

### 预先确定的模型筛选标准

1. 以 **验证 Macro-F1** 为主要排序依据；另记录 Accuracy、中性/悲伤各自召回率、两类最小召回率和三分类混淆矩阵。
2. 候选验证 Macro-F1 至少超过基线 **0.005**。
3. 中性/悲伤两类的较小召回率，不得比基线下降超过 **0.02**。
4. 不满足双重条件时，**保留原 C**，不为凑成果强制更换。
5. 第一轮只是 seed2026 的探索实验。赢家需要第二个 Seed 或独立复现，之后才考虑锁定候选做一次冻结的独立测试、INT8 校准与模拟/上板移交。
6. 不能根据测试集成绩挑实验；不能以手机视觉感受替代正式准确率报告。

## 实际训练位置（独立于原 C）

- 原 C 已完成权重：GitHub Actions Run [38943721697](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38943721697)
- Kaggle C2 notebook：<https://www.kaggle.com/code/tyhuanga/fpga-c2-neutral-sad-experiments>
- C2 Kaggle 预训练基线私有数据集：`tyhuanga/fpga-c2026-locked-baseline`。只包含真正的 C2026 最佳权重、清洗过的训练/验证清单及质量报告；**没有拷贝 test.csv 或测试照片**。
- 首次 C2 GPU 提交 Actions：Run [38995349738](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38995349738)。Notebook 提交成功，随后 GitHub metadata 的严格断言失败；失败不代表 Kaggle GPU 训练失败。
- 独立、**绝不重新提交模型**的 C2 云端状态追踪和结果归档：Run [38995465208](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38995465208)。
- 本轮 Kaggle 脚本在完整执行成功后输出：
  - `results.json`：3 个候选与原基线的 VALIDATION 指标；
  - `A_hard_sampler/best.pt`、`B_focal/best.pt`、`C_distilled/best.pt`；
  - `teacher/best.pt`（仅教学模型，**不能直接部署安路 FPGA**）；
  - `progress.jsonl` 和各实验历史。
- 结果尚未出现时不报告任何 C2 精度提升，也不改变 FPGA 交接时的 C 基线版本。

## 本阶段停止条件

- 如果验证集指标较基线没有明确提升，回到标签审核/训练目标分析，而不是扩大网络。
- 若提高 sad 召回率导致 neutral 召回率明显下跌，应拒绝这个候选。
- 不自动合并未经人工验证的 FER+ 标签映射。
- FPGA 安路板综合、RTL 和 bitstream 由下一阶段/交接人员负责。当前实验输出仍是模型权重、软件结果，不是 FPGA 位流。
