# FER2013 三分类首次自动数据审计结果

**实测来源**：GitHub Actions [FER2013 public-data audit (GitHub CPU) run #38906357819](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/38906357819)（2026-10-16）
**数据集**：Kaggle `msambare/fer2013` 公开图像目录版。
**范围**：仅 `neutral=0`、`happy=1`、`sad=2`。
**执行方式**：GitHub 托管 CPU Runner 下载公开数据集、审计、生成 CSV/JSON；未启动 GPU、未训练模型。
**结果附件**：本次 Run 的 `fer2013-3class-public-audit` Artifacts，包含 `train.csv`, `val.csv`, `test.csv`, `audit.json`，保存 14 天。

## 自动清洗后样本数

| 表情 | Train | Validation | Test | 合计 |
|---|---:|---:|---:|---:|
| Neutral (0) | 4034 | 883 | 1232 | 6149 |
| Happy (1) | 5872 | 1286 | 1773 | 8931 |
| Sad (2) | 3919 | 863 | 1246 | 6028 |
| 总数 | 13825 | 3032 | 4251 | **21108** |

## 自动检查结果与局限

- 损坏/无法读取图片：**0**。
- 具有冲突表情标签的**图片内容 SHA256 值**：**5**。
- 原始训练集内，文件内容与官方测试集相同的图片行：**148**（筛出，不让训练数据直接复制测试图片）。
- 原始训练集中含冲突标签的图片行：**8**（筛出）。
- 保留 FER2013 原始 Test 分区中的无冲突标签图片，从原始 Train 中划分验证集；没有为凑 70/15/15 而重洗官方 Test。
- SHA256 检查仅识别**完全相同的文件内容**，不能排除不同压缩编码、同视频临近帧、同人不同照片和人工标注偏差；用户指导书要求的人工抽查、许可核对和身份来源级防泄漏仍待完成。
- 统计与 manifest 的最终数量来自实际审计日志，**不等于模型准确率**。

## 自动化执行记录

1. 首次通过 `GitHub Actions -> Kaggle Kernels push` 创建了私人 Kaggle 任务 `tyhuanga/fpga-emotion-audit`，但它在 Kaggle 云端启动后报错，错误为 `FileNotFoundError: /kaggle/input/fer2013`（未找到预期的数据挂载目录）。该远程 Kaggle 任务没有完成审计。
2. 改用 `fer2013-audit-cpu.yml`，在 GitHub CPU Runner 上调用 Kaggle 官方 CLI 下载公开数据集，**完整执行成功并存档结果**。
3. 远程 Kaggle 数据挂载仍须修复和验证，才可声称 Kaggle GPU 训练端到端自动化正常。
4. 当前 **P0 自动化审计初步完成，人工抽查/许可与来源组检测未完成；P1 FP32 训练未启动**。

后续建议：优先核对数据集许可与冲突样本，分析三类平衡和 sad/neutral 混淆风险；再配置可核验的 Kaggle GPU 训练工作流。不能将本次图像统计当作已验证模型性能。
