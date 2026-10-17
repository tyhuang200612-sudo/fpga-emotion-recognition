# C8 — 修复 Kaggle 长单元格日志不可见的分阶段训练（2026-10-17）

## 实际状态，禁止混淆

- **C8 代码已创建且 CPU 离线测试已通过**：[GitHub Actions CPU preflight #39015231043](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39015231043)，用于检查 PyTorch 初始预测一致性、固定数据/结构约束以及37单元格中的20个独立五轮训练单元。
- **此文档创建时 C8 Kaggle GPU 尚未启动！** 原 C7 Kaggle内核 `tyhuanga/fpga-c7-gap2-fixed25-deterministic` 仍显示 `RUNNING`，当前不具备可靠的工具接口来准确识别与取消指定Kaggle运行会话。不能用 `kaggle kernels delete` 删除原始实验，也不能冒险根据猜测的sessionID取消别人的内核。
- [安全启动GitHub Actions](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/workflows/kaggle-c8-staged-launch.yml) 每半小时运行一次**短状态核查**：旧C7仍在运行就退出，不会重复提交GPU；旧C7进入COMPLETE/ERROR/CANCELLED后，如果C8仍不存在则只提交一次新的独立C8；随后后续短检查收集已完成的结果并归档。工作流显示Success只表示该轮安全检查成功，不等于C8已提交或训练完毕。
- **不需要长时间运行的 GitHub Runner 等待 Kaggle**，减少私有仓库 Actions 资源消耗。
- 用户可以在 [旧C7 Kaggle页面](https://www.kaggle.com/code/tyhuanga/fpga-c7-gap2-fixed25-deterministic) 的运行会话管理中停止旧任务，让安全门允许下轮短检查自动提交C8；也可以等待C7自然完成后自动提交。切勿关闭浏览器就以为会话停止。
- C8 单独使用新的 **Private Kaggle Notebook**：[`tyhuanga/fpga-c8-stepwise-log-epochs`](https://www.kaggle.com/code/tyhuanga/fpga-c8-stepwise-log-epochs)，代码`tools/train_c8_stepwise.py`，生成器`tools/build_kaggle_c8.py`，CPU测试`tests/test_c8_stepwise.py`，GitHub调度`.github/workflows/kaggle-c8-staged-launch.yml`。

## 为什么原来没日志、这次怎样改变

旧C7的全部100轮都嵌套在**一个Python Notebook代码单元格**中的 `run_c7()`；尽管使用print(flush=True)和私有working目录JSONL文件，Kaggle的CLI `kernels logs/files/output` 在执行期间均返回空白，直到结束才有可能公开。不能用网页显示Running判断训练质量。

C8改为**37个 Notebook cells，20个独立的5-Epoch训练stage cells**：
- 首先一个初始化cell加载已冻结FER2013 train/val、原C2026基线、SHA256验证的C3教师、GPU确定性设置。
- 每个variant+seed单独有开始cell，然后5个训练cells（每个恰好5轮，合计25轮）以及结束cell。两种训练方案 `GAP2_CE_control` 和 `GAP2_C3_KD`，各两个seed `20261021/20261022`，共4组×25轮=**100 epochs**。
- 每一轮逐批次打印批号、损失，并在epoch末打印训练loss、VAL Macro-F1、neutral/sad recall、中性↔悲伤互误判、用时；每5轮结束打印 `C8_STAGE_DONE`，写 `stage_01.pt..stage_05.pt`、`history.json`，训练所有阶段完毕写 `results.json`和`live-progress.jsonl`。
- 原训练模型、损失、训练seed、独立测试集隔离不变；仍是C5 GAP2结构普通CE和冻结C3知识蒸馏的严格对照，固定25轮，要求确定性PyTorch。**不增加模型参数，不自动发布FPGA或手机APK**。
- **可视性重要限制**：Kaggle网页版的活动Notebook在某些运行模式下可以查看已经执行完的代码单元格和输出；但Kaggle的批量commit日志API有可能仍在整个Notebook完成前返回空白。即使拆了cells，也不能保证 `kaggle kernels logs` 实时提供每个epoch。用户如在Kaggle页面看不到实时输出，应等到至少一个阶段完成后再确认平台支持；不能凭空声称实时直播。
- Kaggle任务结束后，GitHub安全短工作流将检验：真实4份`best.pt`、4份每组25轮`history.json`、20个`stage_XX.pt`、exact 100条`C8_EPOCH_END`、20条`C8_STAGE_DONE`、`results.json`、预先设定的双种子升级门槛，归档到名为 `TinyCNN-C8-Staged-GAP2-CE-vs-KD-Fixed25-Logs` 的真实GitHub Actions artifact（保留30天）。
- **没有完成GPU训练前不得填写任何C8验证准确率。** 新的手机/FPGA性能仍需独立验证；以前C2-C保持现役。

## 如果想把真正实时日志完全转出 Kaggle

由于 Kaggle commit API的限制，**完全保证秒级/逐epoch外部实时日志**需要使用另一个支持带身份验证主动上传日志的目的端（例如用户明确授权的私有HTTP服务或拥有写入权限的GitHub凭据，通过Kaggle Secrets存储），或者转为支持GPU交互式输出的平台。目前没有向Kaggle注入Github写权限或把个人服务/令牌加入Notebook；首要安全目标是不泄露密钥，不为了日志侵入汤圆阅读器服务器。
