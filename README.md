# FPGA 表情识别 — Model A（全新三分类）

**当前开发版本统一名称：A。** 输入 48×48 灰度人脸图像，输出：
- **angry / 愤怒 = 0**
- **happy / 高兴 = 1**
- **neutral / 中性 = 2**

这是新的三分类实验，**绝不会将旧 sad（悲伤）图片重命名为 angry**。A 的训练数据来自 Kaggle `msambare/fer2013` 原始愤怒、高兴、中性文件夹，旧 `neutral/happy/sad` 模型、权重和对应类别输出不用于 A。标签描述的是面部表情外观，不能保证一个人的真实心理状态。

## 数据与训练链路

1. [A 数据 CPU 审计](.github/workflows/fpga-a-dataset-audit.yml)：Kaggle [独立私有 CPU 审计任务](https://www.kaggle.com/code/tyhuanga/fpga-a-angry-happy-neutral-dataset-audit)，不消耗 GPU。读取原始 FER2013 文件夹，生成全新 `train.csv`、`val.csv`、`test.csv`、`audit.json`、`quality-report.json`。
2. [A 数据校验及私有发布](.github/workflows/fpga-a-publish-dataset.yml)：真实审计 `COMPLETE` 后检查类别 ID、数量、train/val/test 解码像素重复、文件 SHA256，成功才上传到 [Kaggle 私有 A 清单数据集](https://www.kaggle.com/datasets/tyhuanga/fpga-a-angry-happy-neutral-clean-v1)。**清单仅含 CSV/JSON 元数据，不上传人脸图片，也不修改原图。**
3. [A GPU 训练](.github/workflows/fpga-a-train.yml)：仅在 A 数据审计合格且私有清单准备妥当后，提交 [Kaggle Model A 私有 Notebook](https://www.kaggle.com/code/tyhuanga/fpga-model-a-angry-happy-neutral)。新建并**从头训练**四层 TinyCNN 16/32/64/64，GAP1，FC3，两个独立随机种子（20261021 / 20261022），每个 35 Epoch，共 **70 Epoch**。不使用旧类别模型的任何 checkpoint。
4. [A 日志及结果验收](.github/workflows/fpga-a-results.yml)：每轮训练写入 `live-progress.jsonl` 并打印损失、验证 accuracy、macro-F1、angry/happy/neutral 召回率和三分类混淆矩阵。每 **5 Epoch** 保存一个训练阶段的参数、优化器和随机状态。只有 70 条 `EPOCH_END`、14 条 `STAGE_END`、两个 `best.pt`、两个完整 `history.json`、以及 `results.json` 全部通过验证，才存档并报告新的真实分数。

> Kaggle 有时不公开正在执行的 Notebook stdout 或 working 输出。写入 `live-progress.jsonl` 的日志不等于 GitHub 能立即读到日志。GitHub `success` 仅表示其对应步骤成功，不等于 Kaggle 模型训练已结束。

## 数据隔离与质量

- 新三分类映射固定 `angry=0, happy=1, neutral=2`，原始文件夹不重命名。
- 验证集从 FER2013 官方 TRAIN 划分；官方 TEST 不参加模型选择。
- 对解码到 48×48 的灰度像素做 SHA256：剔除训练侧完全重复、训练/验证与官方 TEST 重复、冲突标签。官方选中类别 TEST 清单保持独立。
- 标记但不擅自删除模糊／低对比图像；**尚不能完全排除同一人及近似图片泄漏**。这需要额外身份信息或人工审核。
- 原先 `neutral/happy/sad` 的 C2-C 验证 macro-F1 **75.16%** 与现在的 `angry/happy/neutral` 是**不同分类任务，不能直接比较成升级提升**。

## 硬件目标

两片安路 HX1P35A FPGA：主 FPGA 拟承担摄像头输入、预处理、传输和 HDMI；从 FPGA 拟承担 INT8 CNN 推理。Model A 继续使用 48×48×1 输入和可面向硬件量化的常规卷积、BatchNorm 和全局平均池化。

**训练完成不代表可以上板。** 还需 INT8 量化准确率、BN 融合、位宽/溢出验证、RTL 黄金向量、综合时序、BRAM/DSP 资源检查和实际摄像头演示。未经验证不填写板上准确率或 FPS。

## 源码与可复现性

- [数据准备：tools/prepare_a_angry_happy_neutral.py](tools/prepare_a_angry_happy_neutral.py)
- [模型与训练：tools/train_model_a.py](tools/train_model_a.py)
- [Notebook 构建：tools/build_kaggle_model_a.py](tools/build_kaggle_model_a.py)
- [数据单元测试](tests/test_a_dataset.py) 和 [模型单元测试](tests/test_model_a.py)
- [GitHub Actions](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions)

旧 C2/C3/C9 等实验留在历史 Git 与 Kaggle 中供追溯，不再作为 A 的训练输入；旧定期监控工作流已停止以减少浪费。完整权重和历史归档在验证新的 A 训练前不做不可逆删除。
