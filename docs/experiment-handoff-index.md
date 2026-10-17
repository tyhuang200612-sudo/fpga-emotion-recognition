# FPGA 表情识别：交接材料索引（持续维护）
建立时间：2026-10-17。GitHub 私有仓库为长期代码与交接索引，实验模型保留在可授权下载的云端工件/私有 Kaggle 中。**不将人脸图片、个人身份信息、公开测试数据、明文密钥上传到仓库。**

## 冻结的原版（不可覆盖）
- 原始 C2026（16/32/64/64，48×48×1，neutral/happy/sad）及量化导出保持不变。
- 原版 C FP32 **独立测试集**准确率 **74.99%**；C INT8 **独立测试集**准确率 **74.88%**（均为既有历史实验）。
- 训练源：<https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38943721697>
- 原 C、C2-C Android 测试包均保留，C2-C 专用 APK 来源：<https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38998523307>。
- **C2-C 手机端误判仍严重**，并不表示同一 FER2013 验证集成绩等于手机准确率。

## C2 验证实验：已结束，仅验证集
- C2 Kaggle 训练与工件 Run <https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38995638992>
- A 困难样本、B focal、C2-C 旧教师蒸馏；实验以 Macro-F1 和中性/悲伤召回为主。
- 原 C VAL Macro-F1 0.74707286；C2-C VAL Macro-F1 约 0.751586；未通过预先设定的 +0.005 门槛。
- 独立 TEST 对 C2 未执行，不允许用原 C 已经使用的 TEST 反复挑模型。

## C3 验证实验：已结束，**完整交接包**
- Kaggle：<https://www.kaggle.com/code/tyhuanga/fpga-c3-camera-domain-teacher>
- GitHub Actions 成功归档：<https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38999204120>
- 归档工件名称：`TinyCNN-C3-PretrainedTeacher-CameraAug-ValidationOnly`，artifact ID **12088004682**。
- 原始压缩包：`C3-final-experiments.zip`（当前对话中有原件备份），SHA256：`beff18bbed1fff3d4314befe78e73af1a8216a6abf95975c92a06bfb47e9d5ed`。
- 压缩包结构：`fpga-fer2013-c3-experiments/pretrained_teacher/best.pt`、`C3_camera_aug/best.pt`、`C3_pretrained_KD/best.pt`、`results.json`、各 history、`progress.jsonl`、Kaggle log。
- **教师** `pretrained_teacher/best.pt` 44,781,953 bytes，SHA256 `ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c`。这是 ImageNet 预训练 ResNet18 经 FER2013 三分类微调的**完整训练态权重**；仅训练用，严禁当作 FPGA 部署权重。
- `C3_camera_aug/best.pt` SHA256 `3bc57bf3417156ecadfacd7d6d5d420aeb104a698a4821a8c9f53604c10262c9`。
- `C3_pretrained_KD/best.pt` SHA256 `316e66216ec071ecab576191563a3f95a2fda669d1d0c6b7864e83770d0ec1ef`。
- `results.json` SHA256 `a67428c3fbfb38075a9a96f3177a1dc1df9ca1b8782197228f27c3145c37292e`。
- 教师 VAL accuracy 0.8026183283，Macro-F1 **0.7857458579**，neutral recall **0.7146144994**，sad recall **0.7334123223**，中性/悲伤互相混淆 **364** 张。
- C3 camera_aug VAL Macro-F1 0.747825409；C3 pretrained_KD VAL Macro-F1 0.747983224；未超过 C2-C，未自动替换部署。
- C3 仅训练/冻结 VAL，`test_evaluated=false`；须独立的真实手机摄像头采样检测性能。

## 云端持久性与恢复
- GitHub Actions artifact 可能过期（C3 工件上传时设为**保留30天**），不要将其视作永久存储。原始 Kaggle Notebook 私有 Output 及当前对话附件也应保留；交接前务必再次核实可下载性、散列值、完整权重。
- C4 工作流将复制 C3 教师至**独立私有 Kaggle Dataset**，用于后续训练复用；仅迁移经过哈希校验的教师 checkpoint，不上传人脸图片。
- 请在正式交接时再次下载并附带完整 zip、GitHub 仓库提交 SHA、实验指标、源代码及 README，不要只给此目录。

## C4 目标（下一阶段）
- 学生 TinyCNNC 仍为 **16/32/64/64**；C3 强教师冻结复用，不扩大部署学生、不重复用测试集选模型。
- 不同温度和权重的蒸馏 + neutral/sad 成对监督；两个 seed 对照，报告两类召回、Macro-F1 及混淆矩阵。
- 只有相对已有 C2-C 的提升具有复现性且不牺牲中性/悲伤平衡，才考虑更新手机测试 APK；正式 FPGA 交接还需 INT8 与 RTL 对照。
