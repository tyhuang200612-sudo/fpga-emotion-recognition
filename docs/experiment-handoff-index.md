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
