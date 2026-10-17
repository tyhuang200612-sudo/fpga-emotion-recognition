# FPGA 表情识别：C5/C6 训练可复现性与数据风险审计

日期：2026-10-17。**本报告是历史训练日志与代码审计，不是新的模型准确率验证，也没有使用用户手机人脸照片、冻结官方TEST集或启动额外测试集调参。**

## 原始来源与代码
- C5 GPU [Actions #39004573026](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39004573026)，真实原始artifact ID `12090472500`，SHA256 `19f43cc26cdef6c308d9fbfafcb6b383b3f522823356b7b51ab74416369a0055`。
- C6 GPU [Actions #39008401226](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39008401226)，真实原始artifact ID `12091822756`，SHA256 `1e3d042a15c4980bb65675cd6cd515ad957fd70384353b52a7ef3a816015d241`。
- 两次归档均包含分seed逐轮 `history.json`、报告 `results.json` 及 best 权重。独立审计源代码 `tools/audit_c5_c6_repro.py`；GitHub C7 工作流在GPU提交之前直接下载原始二进制归档、校验这两个SHA并重新运行审计。
- 训练源代码：`tools/train_c5.py`、`tools/train_c6_spatial_kd.py`。C5 的 `GAP2_spatial` 与 C6 的 `GAP2_CE_control` 仅用于检查相同种子、相同卷积/池化头、相同数据、加权CE、学习率和early-stop下的历史重复性，不能把它们当作完全独立的新模型验证。

## 最关键的真正发现：原早停算法对微小差异非常敏感

| 指标 | C5 GAP2 seed 20261022 | C6 GAP2 CE seed 20261022 |
|---|---:|---:|
| 已执行epoch总数 | **25** | **8** |
| 第1轮验证 Macro-F1 | 0.74164731 | 0.74226248 |
| 第8轮验证 Macro-F1 | 0.741733（近似） | 0.74169（近似） |
| 选择的最佳epoch | **21** | **1** |
| 历史最好 Macro-F1 | **0.75206811** | **0.74226248** |

- C5 第8轮 F1 相比第1轮只高约 **0.000086**，即约 **0.0086个百分点**，刚好刷新了最佳纪录，因此重置没有提升的等待计数器；接下来第12、15、16、21轮有进一步改进机会，最终达到0.75206811。
- C6 同seed第8轮接近但没有超过第1轮的0.74226248，所以累积“连续7轮未提升”，第8轮提前结束，从根本上失去后续17轮的训练机会。这是**导致两个存档所报告最好分数差异被放大的一条可复现的程序路径**。
- C5/C6 同seed前8轮平均训练loss绝对差仅约 **0.0000378064**，对应平均F1绝对差约 **0.00083619**。另一个种子20261021共同15轮的平均loss绝对差仅约 **0.0000451134**，F1平均差约 **0.00115835**。
- 这与GPU数值/随机非确定性造成细微扰动、随后early-stop放大结果**相容**，但不能只凭历史归档断定细微扰动的唯一来源就是 cuDNN。也没有证据说明C5/C6训练数据变了或发生训练集泄漏。

## 训练数据审计边界

- 两轮源代码核对了清洗策略 `fer2013-three-class-safe-pixel-dedup-v1`、13,571 train和2,979 VAL，并一致锁定原始官方 TEST manifest SHA256 `d2ed56d7cb6f6c6d38ee24c8acabb543fb1b8a3ce2af68f3f03ec2d88051e3ac`，所有训练报告记为 `test_evaluated=false`。
- `tools/prepare_fer2013.py` 和 `tools/optimize_fer2013_for_c.py` 仅防相同字节、完全相同解码像素造成的明确重复/矛盾标签泄漏；**尚无人物身份、相似人脸跨组的可靠排除，不能声称“按人完全独立”**。
- 该任务仅neutral/happy/sad三分类。冻结VAL有中性869张、开心1266张、悲伤844张。现役 C2-C Macro-F1约75.16%，对中性/悲伤召回仍明显弱于开心；模型结构容量、低分辨率数据、标签歧义和真实手机domain shift都可能共同造成性能上限，无法凭本审计判断是哪一项单独主导。
- 用户此前提供的65张连续手机裁剪，只有19张悲伤标签得到用户确认；46张中性/开心标签因为未及时切换，**不能当有监督准确率证据或自动修正标签**。本次审计与C7均不触碰任何手机原图。

## 采用的修复、验证规则和后续限制

- `tools/train_c7_fixed25.py`：强制 `C6_EPOCHS=25` 与 `C6_PATIENCE=26`，即两种训练方式的两个随机种子**都完整跑满25轮**；每轮保存验证记录，仍按预设Macro-F1选择最佳权重，但不因随机波动提前停止另一组训练。
- 启用 `torch.use_deterministic_algorithms(True,warn_only=False)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；如GPU不支持确定性运算则直接报错，不悄悄退化为不确定性模式。
- `tools/build_kaggle_c7.py` 将 C7 训练放到新的Kaggle私有Notebook `tyhuanga/fpga-c7-gap2-fixed25-deterministic`，不覆盖C5/C6，也不改C3教师和训练/验证划分。对照仍为 `GAP2_CE_control` vs `GAP2_C3_KD`，两种种子各完整25轮；模型相同，仅损失的蒸馏项不同。
- GPU 工作流：[GitHub Actions #39010774856](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39010774856)。GitHub在启动GPU前已运行完整历史归档哈希检查、early-stop审计并通过；若最终成功还须验证**四份权重、100条 EPOCH_END、每组25轮历史**及冻结教师checkpoint哈希。
- **重要**：若C7再次未过双seed可靠提升门槛，应停止在相同FER2013 VAL上反复选择，并优先构建按人、拍摄会话隔离的真实手机评测集。任何结果都不是FPGA上板INT8准确率：新模型没有被量化、合成RTL、烧录或实测。不能自动替换现役C2-C。
