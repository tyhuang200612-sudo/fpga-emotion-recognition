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


## C4 冻结教师双随机种子蒸馏：已结束，未达到发布门槛（2026-10-17）
- 成功运行：<https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39001287016>；Kaggle 私有 Notebook：<https://www.kaggle.com/code/tyhuanga/fpga-c4-balanced-knowledge-distill>。
- GitHub 工件：`TinyCNN-C4-Frozen-C3-Teacher-Two-Seed-ValidationOnly`，artifact ID **12089304028**，含两种方案×两个随机种子的4份 `best.pt`、训练历史、`results.json`、`progress.jsonl` 与 Kaggle log。
- 在此对话已下载完整 `C4-final-validated-results.zip`，SHA256：**9a40aff20b9320c38ec5a62b0d15ecd319ca8633164e4701cdc4666891623d1f**，解压后检查 ZIP 完整性通过（11个条目）。
- 冻结 C3 教师 VAL Macro-F1 **0.785745858**，中性/悲伤互相混淆364；C4 四次运行均复用了精确 SHA256 `ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c` 的教师权重。
- 原版 C VAL Macro-F1 **0.747072860**，中性/悲伤互相混淆417；历史 C2-C VAL Macro-F1 **0.751586**，中性/悲伤互相混淆408。
- C4 `C4_pairwise`：
  - Seed 20261019：epoch 1；accuracy 0.758308；Macro-F1 **0.740873**；neutral recall **0.627158**；sad recall **0.741706**；中性/悲伤互相误判 **421**。
  - Seed 20261020：epoch 14；accuracy 0.762001；Macro-F1 **0.745980**；neutral recall **0.639816**；sad recall **0.753555**；互相误判 **413**。
  - 两种子平均 Macro-F1 **0.743426**，平均互相误判 **417**。
- C4 `C4_margin`：
  - Seed 20261019：epoch 1；accuracy 0.758644；Macro-F1 **0.740944**；neutral recall **0.630610**；sad recall **0.733412**；互相误判 **414**。
  - Seed 20261020：epoch 23；accuracy 0.763679；Macro-F1 **0.744613**；neutral recall **0.655926**；sad recall **0.694313**；互相误判 **405**。
  - 两种子平均 Macro-F1 **0.742779**，平均互相误判 **409.5**。
- `robust_winner_by_validation = none`，**C4 没有发布到 Android，也没有覆盖原版 FPGA 权重**；`test_evaluated=false`，`phone_camera_evaluated=false`。
- 下一轮应该先做诊断：控制单变量剥离 C4 的手机增强/强中性悲伤损失/置信度门控；评估小学生可达到的上界与域差异；需要独立、知情采集的手机真实表情数据评测才能宣称手机性能提升。禁止仅通过选阈值或不断刷冻结 VAL 来制造提升。

## 交接备份状态提醒
- 完整本地交接包 `FPGA_Emotion_C2_C3_Handoff_2026-10-17.zip` 已存在当前对话的下载资产中，含 `C2_results.zip`、`C3_results.zip`、`manifest.json`，本地 SHA256 **597a2bdbaa8504bf96ddf9302a1f228fdeb37a4d0d4de2b43b57deaf90e493e5**。
- 试图创建 Kaggle 私有备份 `tyhuanga/fpga-emotion-handoff-c2-c3` 的 GitHub Action **39001467186 为 failure**：上传命令返回成功，但超过等待期仍无法验证数据集文件列表。**私有 Kaggle 交接备份的可下载性尚未确认，不能标记“已永久备份成功”**。后续交接须核实这一点。
- C4 独立模型归档工件只有 30 天保留期；正式交接前需要下载/核验并合并到最新交接包，或写入经过验证的私有持久备份。


## C5 空间池化结构消融（2026-10-17）
- 设计与测试文档：[C5 GAP1 vs GAP2 验证协议](c5-spatial-ablation-protocol.md)；对照组维持原 TinyCNN-C 四层 16/32/64/64；实验组仅末端 1×1 GAP→2×2 GAP，FC权重增加576（60,240→60,816，不含BN/偏置）。
- 资源暂是**理论估计，不是安路 FPGA 面积/时序、INT8 精度结论**；测试证明扩展头从C2026权重初始化后在训练前与原C输出数值近似一致。
- 验证独立工作流：[GitHub Actions C5 #39004573026](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39004573026)，Kaggle私有 notebook [fpga-c5-gap1-vs-gap2-ablation](https://www.kaggle.com/code/tyhuanga/fpga-c5-gap1-vs-gap2-ablation)。
- 对照两种结构×两个 seed：20261021、20261022；相同13,571 train +2,979 val数据、相同增强训练、参数和原版C初始化，不读取冻结官方TEST或手机私有样本，不重新训练教师。
- GitHub提交源码：`src/model_c5.py`、`tools/train_c5.py`、`tools/build_kaggle_c5.py`、`tests/test_c5_spatial.py`、`.github/workflows/kaggle-c5-spatial-ablation.yml`。
- **截止编写时状态：GPU运行中**，未产生可验证的C5成绩/权重，不能声称已改善或已可部署；归档工件名预留 `TinyCNN-C5-GAP1-vs-GAP2-TwoSeeds-ValidationOnly`，仅任务成功后才能引用下载和校验哈希。
- 本轮无用户人脸图提交到GitHub；`docs/camera-teacher-student-diagnostic-2026-10-17.md` 仅为不含照片和身份的汇总。


## C5 最终训练成绩（2026-10-17；冻结VAL，仅结构消融）
- GitHub 训练归档：[Actions #39004573026](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39004573026)；Kaggle 私有 Notebook：`tyhuanga/fpga-c5-gap1-vs-gap2-ablation`；GitHub Artifact 名 `TinyCNN-C5-GAP1-vs-GAP2-TwoSeeds-ValidationOnly`，artifact ID **12090472500**，压缩包名 `C5-GAP1-vs-GAP2-ValidationOnly.zip`，SHA256 **19f43cc26cdef6c308d9fbfafcb6b383b3f522823356b7b51ab74416369a0055**。
- 原版 C 验证 Macro-F1 **0.747072860**，中性↔悲伤误判 **417**；既有 C2-C 参考 Macro-F1 **0.751586412**、误判 **408**。
- GAP1 控制组（和 GAP2 同配置、相同 seed）：
  - seed 20261021 epoch 8：accuracy **0.763343404**、Macro-F1 **0.746861493**、neutral recall **0.645570**、sad recall **0.742891**、中性↔悲伤互相误判 **411**。
  - seed 20261022 epoch 1：accuracy **0.757972474**、Macro-F1 **0.741207168**、neutral recall **0.600690**、sad recall **0.787915**、互相误判 **423**。
  - **平均 Macro-F1 0.744034330，平均互相误判417**。
- GAP2 空间池化实验组：
  - seed 20261021 epoch 8：accuracy **0.765693186**、Macro-F1 **0.748928014**、neutral recall **0.668585**、sad recall **0.715640**、互相误判 **407**。
  - seed 20261022 epoch 21：accuracy **0.770392749**、Macro-F1 **0.752068108**、neutral recall **0.651323**、sad recall **0.722749**、互相误判 **395**。
  - **平均 Macro-F1 0.750498061，平均互相误判401**。相较本次 GAP1 控制组 Macro-F1 **+0.006463731**（+0.646 个百分点）、平均误判 **-16**；但比历史 C2-C 参考分数仍低约 **0.001088351**（-0.109 个百分点）。
- **资格判断：`gap2_qualified_for_independent_phone_test=false`，两个 seed 均未通过预设的“对 C2-C 至少 +0.005 Macro-F1、误判至少少10、两类召回不明显下降”的全部要求。中性召回约0.6686与0.6513仍低于C2-C约0.6824；悲伤召回0.7156与0.7227也低于C2-C约0.7299。**
- 结构理论资源：60,240→60,816 Conv+FC int8 权重（增加576），无实际 RTL、BRAM、时序、INT8 新模型对照。训练结果 `test_evaluated=false`、`phone_camera_evaluated=false`、`no_automatic_deployment=true`。**不得直接替换原 FPGA C、手机 C2-C 或宣传手机场景已变好**。
- 归档含 4 份 `best.pt`、4 份 `history.json`、`results.json`、进度日志和 Kaggle log；交接时连同 C2/C3/C4 旧包以及 manifest 合并，并核验 SHA256。工作流工件有有效期，请避免过期后唯一副本丢失。
- **下一步**：这次针对 GAP 的证据是“优于同期对照，但未达历史最佳”。若继续研发，建议在固定结构下开展 GAP2 学生知识蒸馏 vs GAP2 普通训练的严格同配置对照，并分离新手机用户/场景测试集；不要直接用已反复观察的19张连续悲伤图选模型。

## C6 GAP2 教师蒸馏对照（2026-10-17，正在训练）
- [实验协议](c6-gap2-distillation-protocol.md)。固定 C5 GAP2 结构（Conv通道16/32/64/64、2×2池化、FC256→3），从冻结C2026权重精确初始化。
- 对照 `GAP2_CE_control`（普通加权交叉熵）与 `GAP2_C3_KD`（80%加权CE + 20%冻结C3强教师KL，T=2），两个配对随机种子 20261021/20261022，全部相同 FER2013 train/val和训练设置。
- Kaggle私人模型：[fpga-c6-gap2-frozen-c3-kd](https://www.kaggle.com/code/tyhuanga/fpga-c6-gap2-frozen-c3-kd)，GitHub Actions [#39006028260](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39006028260)。
- GPU私有数据集包含C3冻结ImageNet预训练教师，训练脚本将核对教师SHA256、验证集分数与样本数量。原独立TEST不参与模型选择，不使用用户手机诊断原图，不自动改表情标签。
- 预期工件 `TinyCNN-C6-GAP2-CE-vs-C3Teacher-KD-TwoSeeds-ValidationOnly`，**此条记录写入时训练仍在进行，尚无可信最终成绩**；收到完整归档后必须检查4份最佳模型、分种子精度、双向混淆数、门槛、SHA256、有效期并加入交接包。
- 禁止C6训练后直接替换Android APK/FPGA权重；尚未量化或完成安路RTL上板验证。


## C6 手动取消后的日志增强重训（2026-10-17）
- 原始 C6 private Kaggle `tyhuanga/fpga-c6-gap2-frozen-c3-kd` 由用户主动结束，原始 GitHub Actions [#39006028260](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39006028260) 最终 **failure**。原运行未归档四个最佳权重，**不得声称其验证成绩**。
- 新的独立 Kaggle Notebook：[`tyhuanga/fpga-c6-gap2-kd-verbose-retry1`](https://www.kaggle.com/code/tyhuanga/fpga-c6-gap2-kd-verbose-retry1)。自动训练与监控：[GitHub Actions #39008401226](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39008401226)。新内核独立，不覆盖已停止的旧核。
- 新增 `tools/build_kaggle_c6_verbose.py`、`tests/test_c6_verbose.py`、`.github/workflows/kaggle-c6-verbose-retry.yml`。复用 `tools/train_c6_spatial_kd.py` 且加入 `c6_progress()`：保存到私有 Kaggle `/kaggle/working/fpga-fer2013-c6-gap2-kd-verbose-retry1/live-progress.jsonl`，同时 `print(...,flush=True)`。
- 日志节点包括 START、数据集扫描、冻结教师校验、EPOCH_START、每25批一次BATCH、VALIDATION_START、EPOCH_END（包含train loss、accuracy、macro F1、中性/悲伤召回率、完整混淆矩阵）、VARIANT_END、COMPLETE，并在异常时打印 traceback。
- 工作流每40秒读取 Kaggle 任务状态，每4轮尝试通过 `kaggle kernels logs` 提取 C6_PROGRESS；**Kaggle在运行中的长代码单元可能推迟向网页/API提供stdout，不保证实时显示**。任务完成后 `live-progress.jsonl`、`progress.jsonl`、`history.json`、`results.json` 与4个 `best.pt` 才是可复核证据。新run启动检查全部通过并已推送GPU Notebook，但记录创建时 **GPU尚未返回正式C6质量报告**。
- 保持原有独立TEST及手机人脸诊断样本隔离，不自动改标签、不部署Android和安路FPGA、不声称C6质量提升。


## C6 日志增强重训已完成（2026-10-17）

- **最终状态：GitHub Actions [#39008401226](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39008401226) Success**；独立私有 Kaggle `tyhuanga/fpga-c6-gap2-kd-verbose-retry1`。
- 真实成绩与逐seed混淆矩阵：[C6 最终完整验证报告](c6-final-verified-results-2026-10-17.md)。
- 模型归档 GitHub Artifact `TinyCNN-C6-Verbose-Retry1-GAP2-CE-vs-C3KD-TwoSeeds`（ID `12091822756`），下载归档ZIP文件 `C6-verbose-validated-GAP2-CE-vs-C3KD.zip`，**SHA256 `1e3d042a15c4980bb65675cd6cd515ad957fd70384353b52a7ef3a816015d241`**。
- 数据完整：4份最佳 `best.pt`，4份 `history.json`，1份 `results.json`、结构化 `live-progress.jsonl`（809条，其中66个`EPOCH_END`、594个`BATCH`）、`progress.jsonl`和Kaggle运行日志。
- C6-C3KD双seed Macro-F1 **74.9325%**、中性↔悲伤混淆平均 **401**，对应同期GAP2-CE双seed **74.4751%**、混淆平均 **412.5**；虽然蒸馏平均提升 **0.4574 个百分点**、混淆少 **11.5**，但**没有超过历史 C2-C 75.1586% 的F1，两个seed均未通过可靠升级门槛**（`robust_improvement_gate=false`）。
- 原用户手机19张近重复悲伤图没有用于训练或本实验验证。独立 TEST 未触碰。尚无手机/FPGA INT8部署。保留原C2-C为现役。
- **注意**：同设置同期CE与历史C5 GAP2分数不同，表明随机数控制/复现性等还有待审计；不能宣称蒸馏效果在独立实验中稳定。


## C7 训练可复现性审计与固定25轮实验（2026-10-17）
- [独立诊断与数据风险报告](c7-reproducibility-data-audit.md) 通过原始C5 artifact **12090472500** (SHA256 `19f43cc26cdef6c308d9fbfafcb6b383b3f522823356b7b51ab74416369a0055`) 和C6 artifact **12091822756** (SHA256 `1e3d042a15c4980bb65675cd6cd515ad957fd70384353b52a7ef3a816015d241`) 复核。
- **明确定位到早停敏感性**：同seed20261022，C5 GAP2在epoch8因相对epoch1微小Macro-F1增长（约0.000086）重置早停等待，完成25轮、最佳epoch21 F1=0.752068；C6同期GAP2普通训练在epoch8早停、最佳epoch1 F1=0.742262。这解释两轮结果差距被早停放大，但细微GPU差异的唯一来源尚未证实。
- 代码修复和审计：`tools/audit_c5_c6_repro.py`、`tools/train_c7_fixed25.py`、`tools/build_kaggle_c7.py`、`tests/test_c7_fixed25.py`、`.github/workflows/kaggle-c7-deterministic.yml`。独立审计已通过GitHub实际原档下载、哈希与训练/验证划分校验。
- C7固定同一TinyCNN-C 16/32/64/64 GAP2、C3强教师SHA `ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c`，对照CE与CE(80%)+KD(20%,T=2)，各两seed **不早停而完整训练25轮**，强制strict deterministic CUDA并保存全部100轮epoch日志。
- [GitHub C7 实际GPU运行 #39010774856](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39010774856)，[Kaggle私有C7 Notebook](https://www.kaggle.com/code/tyhuanga/fpga-c7-gap2-fixed25-deterministic)。启动前数据审计、C7 Notebook生成测试、教师挂载和GPU提交已成功，**本条写入时仍在训练中，尚未得到最终成绩**。
- 预期artifact名 `TinyCNN-C7-Fixed25-Deterministic-GAP2-CE-vs-KD`，必须在运行成功且核验四份模型、每份完整25轮、100条`EPOCH_END`及`results.json`后才承认实验完成。现役仍为C2-C。原独立TEST、私有手机诊断图和安路真实上板测试均未进入C7。

## C7 无实时日志排查（2026-10-17）

- 真实 GitHub 训练：[Actions #39010774856](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39010774856)；Kaggle 私有内核 `tyhuanga/fpga-c7-gap2-fixed25-deterministic`。**任务提交成功，但在无外部epoch记录前不能称为已确认正常执行模型。**
- 独立三次只读诊断：[第一次 #39012531459](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39012531459)、[第二次 #39014133032](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014133032)、[源码+文件API深度检查 #39014326452](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014326452)。Kaggle反复返回`RUNNING`，`kaggle kernels logs` 仅1字节，运行中 `files/output` 未暴露进度文件；已下载实际8-cell内核源码，并核实末单元`run_c7()`和固定25轮设置。
- 根因归类：**确定存在可观测性缺陷**：C7整轮100个Epoch在一个长Notebook单元中执行，本地`live-progress.jsonl`无法通过Kaggle当前实时输出接口读取，`c6_progress()`未上报远端Epoch。**尚不能证实GPU实际已跑多少轮、是否卡死。**
- [根因报告](c7-live-log-forensic-investigation.md)。新增备用[按需归档工作流](../.github/workflows/kaggle-c7-collect-on-demand.yml)，只有Kaggle运行`COMPLETE`时才下载并核验四份权重+100轮，绝不重训；该工作流仅允许手动触发，避免重复GPU计算。
- 当前长时间GitHub轮询工作流仍可能计入私有仓库Actions分钟。后续训练应改为“提交即返回 + 短时定期检查 + 完成后短时归档”，避免持续占用Runner，不应为查日志反复取消和重训。


## C8 分阶段日志重训（2026-10-17，准备完成、GPU待安全启动）
- 用户要求重新训练且能查看日志。此前 C7 均采用一个长训练cell，Kaggle `kernels logs/files/output` 运行期间均返回空白，未得到任何实际Epoch证据。
- C8 新源代码 `tools/train_c8_stepwise.py`、`tools/build_kaggle_c8.py`、`tests/test_c8_stepwise.py`。已通过 [离线CPU检查 #39015231043](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39015231043)：37个独立Notebook代码/元单元、**20个每5 Epoch结束的训练cell**，共4份模型×25轮，每轮打印、写 `live-progress.jsonl`；每5轮保存 `stage_XX.pt`、`history.json`，最后保留 `best.pt`与`results.json`。
- [完整说明](c8-staged-kaggle-logging-and-safe-relaunch.md)。
- GitHub短工作流 `.github/workflows/kaggle-c8-staged-launch.yml` **每30分钟核实**原C7与新C8 Kaggle状态。原C7 `RUNNING`时**坚决不启动第二份GPU**；C7结束、C8不存在后提交一次新Kaggle私有kernel `tyhuanga/fpga-c8-stepwise-log-epochs`；C8完成后再次运行短任务校验100个训练事件与4份权重、上传 `TinyCNN-C8-Staged-GAP2-CE-vs-KD-Fixed25-Logs` 工件。此记录创建时 **C8 尚未启动GPU**，不得将GH安全检查Success当训练成功。短检查替代GitHub Runner长时间占用等待。
- Google Kaggle当前CLI版本的 `kaggle kernels push --no-run` 不可用，故私有无GPU预览若不支持则安全地跳过，改成下载经本地测试的 Notebook GitHub artifact `C8-Staged-37-Cell-Kaggle-Notebook-Source`，由 `.github/workflows/kaggle-c8-private-preview.yml` 发布，**不会意外触发GPU训练**。
- 为节省Kaggle GPU配额，用户若希望立即开始C8，需要先在Kaggle旧C7运行会话操作Stop，或等原C7自然结束；一旦C7停止，自动安全启动可能延迟最多约30分钟（受GitHub Actions schedule排队影响）。
- **重要限制**：拆分Notebook cells提高Kaggle活动页面可观察性，但当前Kaggle CLI API不保证训练结束之前提供日志或阶段文件；不能承诺外部API实时逐Epoch直播，最终完成后至少完整提供100条Epoch、20条阶段和20份stage权重用于审计。
- 未触碰独立TEST，未上传用户人脸图片或错误标签，未自动替换C2-C、Android APK或FPGA ROM；不能预先填报C8验证/手机准确率。
