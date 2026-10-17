# C6 详细重训最终验证报告（2026-10-17）

**状态：已完整完成，真实 GitHub Actions Success；不符合模型替换门槛。**

## 实验来源和验证文件

- GitHub Actions [C6 verbose retry #39008401226](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39008401226)：状态 **completed/success**，2026-10-17T06:20:37Z。
- Kaggle 私有 Notebook：[`tyhuanga/fpga-c6-gap2-kd-verbose-retry1`](https://www.kaggle.com/code/tyhuanga/fpga-c6-gap2-kd-verbose-retry1)。
- GitHub Actions artifact：`TinyCNN-C6-Verbose-Retry1-GAP2-CE-vs-C3KD-TwoSeeds`，ID **12091822756**；获取的完整 ZIP：`C6-verbose-validated-GAP2-CE-vs-C3KD.zip`，大小约958 KiB、**SHA256 `1e3d042a15c4980bb65675cd6cd515ad957fd70384353b52a7ef3a816015d241`**。
- 归档含 `results.json`、四份 `best.pt`、四份 `history.json`、`progress.jsonl`、`live-progress.jsonl` 以及Kaggle运行日志。解析 `live-progress.jsonl` 共 **809 条事件**，其中 `EPOCH_END` **66 条**，`BATCH` **594 条**，`VARIANT_START/END` 各4条、完整 `COMPLETE` 一条。
- 源代码：`tools/train_c6_spatial_kd.py`、`tools/build_kaggle_c6_verbose.py`、`tests/test_c6_verbose.py`、`.github/workflows/kaggle-c6-verbose-retry.yml`。
- 独立_TEST_ 未参与；手机用户照片未参与；教师 checkpoint SHA256：`ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c`。FER2013 frozen cleaned train 13,571 / VAL 2,979。

## 与同期控制组的严格对照

| 训练条件 | Seed | Best epoch | 验证 Accuracy | 验证 Macro-F1 | 中性召回 | 开心召回 | 悲伤召回 | 中性↔悲伤误判 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GAP2 + 加权交叉熵 | 20261021 | 8 | 76.4015% | 74.7239% | 66.8585% | 86.4139% | 71.2085% | 409 |
| GAP2 + 加权交叉熵 | 20261022 | 1 | 75.9651% | 74.2262% | 60.5293% | 85.4660% | 77.6066% | 416 |
| GAP2 + 冻结C3教师 KD (20%, T=2) | 20261021 | 11 | 76.4015% | 74.6177% | 70.4258% | 87.2828% | 66.2322% | 410 |
| GAP2 + 冻结C3教师 KD (20%, T=2) | 20261022 | 21 | 77.1064% | 75.2473% | 64.3268% | 88.6256% | 72.9858% | 392 |

| 汇总 | 平均 Macro-F1 | 平均中性↔悲伤误判 |
|---|---:|---:|
| C6 同期 GAP2 CE 控制 | **74.4751%** | **412.5** |
| C6 GAP2 + C3 KD | **74.9325%** | **401.0** |
| 差值 KD − CE | **+0.4574个百分点** | **−11.5** |
| 历史 C5 GAP2，两seed均值 | **75.0498%** | **401.0** |
| 现役 C2-C 验证结果 | **75.1586%** | **408.0** |

**结论：** C6蒸馏在本轮同期控制组上得到小幅平均提升，但不能称为可靠突破：种子20261021的蒸馏 Macro-F1 比同期CE稍低，悲伤召回从71.21%降到66.23%；种子20261022的中性召回只有64.33%，低于历史C2-C约68.24%。结果中 `seed_gates={"20261021":false,"20261022":false}`，`robust_improvement_gate=false`。

## 完整混淆矩阵

类别顺序：neutral、happy、sad；**矩阵按真实类别为行、预测类别为列**。

```text
C6 CE seed 20261021:
[[581,  76, 212],
 [ 90,1094,  82],
 [197,  46, 601]]

C6 CE seed 20261022:
[[526,  76, 267],
 [ 72,1082, 112],
 [149,  40, 655]]

C6 KD seed 20261021:
[[612,  75, 182],
 [ 93,1105,  68],
 [228,  57, 559]]

C6 KD seed 20261022:
[[559,  89, 221],
 [ 68,1122,  76],
 [171,  57, 616]]
```

## 限制和决策

1. C6 控制组的两seed平均 Macro-F1 **74.4751%** 与此前 C5 GAP2实验 **75.0498%**（尽管设置和种子标称相同）有差异；需要查清计算非确定性、数据批次增强随机源、日志/构建版本和初始化的可复现因素。在消除变异前，不应将平均+0.457个百分点当成稳健确定的蒸馏收益。
2. 仍沿用反复进行模型选择的同一 FER2013 VAL；这不是新的独立真实手机评测。此前19张连续悲伤照片仅用于问题诊断，并非本轮训练、调参或性能证明。
3. `test_evaluated=false`、`phone_camera_evaluated=false`、`int8_fpga_verified=false`、`no_deployment=true`；无新FPGA ROM/Verilog/bitstream，也没有新手机APK部署。
4. **现役模型继续保留 C2-C**。不因某次 C6 种子报告 75.2473% 就视为新模型合格。下一步优先进行可复现性排查、独立人/场景手机测试和真正的FPGA硬件准备，而不是无限在同一验证集上试验。
5. GitHub Actions artifact 保留期30天，**需要下载并永久备份**；仓库只存隐私安全的实验摘要、不上传原始人脸图片，也不把二进制checkpoint直接写入GitHub源码目录。
