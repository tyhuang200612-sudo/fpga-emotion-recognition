# 双 FPGA 视觉表情识别系统

**Tiny-CNN-A + INT8 + 时序平滑** | 面向双安路 HX1P35A（PH1P35MDG324）。

此仓库依据用户提供的《双 FPGA 视觉表情识别系统实验指导书 V1.0》开发。所有模型准确率、功耗、资源占用、帧率都是**待测指标**，不得提前填成实测值。

## 目标与硬件

- 分类：**neutral（中性）/ happy（高兴）/ sad（悲伤）**，并不代表真实心理状态。
- 输入：**48×48×1 灰度图**。
- 网络：Conv(8) → Pool → Conv(16) → Pool → Conv(32) → GAP → FC(3)。
- 训练：FP32 PyTorch → INT8 PTQ，必要时 QAT；最终使用黄金向量与 RTL 逐层对照。
- 主板：摄像头 / ROI 与缩放 / SPI / 时序投票 / HDMI。
- 从板：SPI / 缓冲 / **真实** INT8 CNN RTL 推理。
- 主从通信：拟使用 SPI + 控制信号，待核对原理图。

## 当前状态

- [x] GitHub 私有仓库与 Colab 笔记本已建立。
- [x] Tiny-CNN-A 架构源码、三分类配置、CPU 单元测试已入库。
- [x] GitHub Actions 自动化基础验证工作流。
- [ ] 数据集审计、按人/视频分组及无泄漏划分。
- [ ] Colab T4 正式训练 / 验证 / 三随机种子。
- [ ] PTQ / QAT、定点参考与 golden 测试向量。
- [ ] FPGA RTL、综合与双板实测。

> **注意**：GitHub Actions 中的普通 `ubuntu-latest` 运行器不会分配你的私人 Google Colab T4。免费 Colab 也不适合充当远程可控 Actions Runner。Actions 目前只做 CPU 代码检查，训练尚未启动。详见 [自动化说明](docs/actions-and-colab.md)。

## 在线使用

- [Colab 笔记本（需要对私有仓库有访问权限）](https://colab.research.google.com/github/tyhuang200612-sudo/fpga-emotion-recognition/blob/main/colab_setup.ipynb)
- [GitHub Actions](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions)

## 目录

```text
.github/workflows/fpga-ci.yml   # GitHub Actions: CPU 结构测试
colab_setup.ipynb               # Colab 笔记本（尚未用于训练）
configs/train_v1.yaml           # 按实验指导书的建议参数
src/model.py                    # Tiny-CNN-A 网络
tests/test_model.py             # CPU 前向/参数数目验证
tools/validate_notebook.py      # 检查 Notebook 格式与泄漏输出
docs/actions-and-colab.md       # Actions 与 Colab 能力边界
```

## 参数预算核对

指导书中的卷积权重加法存在一处笔误：72+1152+4608=**5832**，加 FC 的 96 个权重后是 **5928**，不是文中的 6072。另加 BN 可训练参数 112 和 FC 偏置 3，Tiny-CNN-A 可训练参数合计 **6043**。这不改变设计的约 1.49 M MAC/帧估计。

## 安全性

原始人脸图像、训练集、模型权重和个人身份信息默认不提交 GitHub。数据集需要核验合法来源、标签定义、许可与数据泄漏风险。训练与推理报告仅在完成实测后填写。
