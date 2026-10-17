# C7 Kaggle 训练无可见日志：实际取证结论与后续修复（2026-10-17）

## 已经核实的事实

- Kaggle 私有内核：`tyhuanga/fpga-c7-gap2-fixed25-deterministic`。真正训练的 GitHub Actions：[运行 #39010774856](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39010774856)。原工作流成功提交 notebook 并验证了 GPU 配置与私有数据挂载，随后进入长时间的 `Track C7 fixed25 GPU progression and retrieve available logs` 轮询。
- 第一轮直查：[只读 #39012531459](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39012531459)，Kaggle 回 `KernelWorkerStatus.RUNNING`，`kaggle kernels logs` 返回 **1字节**；无验证 epoch。
- 第二轮直查：[只读 #39014133032](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014133032)，结果同上。
- 深度取证：[只读 #39014326452](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014326452)：Kaggle仍返回 `RUNNING`；`kaggle kernels logs` 仅1字节，`kaggle kernels files --page-size 50` 无输出条目，`kaggle kernels output --file-pattern live-progress.jsonl` 当前没有可下载数据。**这些接口没有暴露正在运行时的工作目录/实时stdout，不能据此声称进程卡死。**
- 同次深度取证通过 `kaggle kernels pull -m` 下载并检查真正提交的**8单元格 Notebook**：训练源码确实存在 `run_c7()`、`C7_EPOCHS=25`、`torch.use_deterministic_algorithms(True,...)`、`c6_progress("EPOCH_END",...)`、`C6_PATIENCE=C7_EPOCHS+1`。**最后一个单元格确实调用 `run_c7()`**。但是这只证明远程代码与GitHub预期一致，不能单凭静态代码证明训练函数已经执行。
- C7 训练入口在**最后一个单元格内一次性运行4个学生任务（CE种子20261021/20261022，KD种子20261021/20261022），每个25轮，共100轮**；中间 `c6_progress()` 一方面 `print(...,flush=True)`，一方面把事件追加到容器内 `/kaggle/working/.../live-progress.jsonl`。Kaggle输出目录未通过当前公共API在运行中暴露，故GitHub通过 `kaggle kernels logs` 只能等而看不到 epoch。
- 虽然历史生成器沿用了 `tools/telemetry.py` 的初始化连通性检查，C7所实际执行的 `c6_progress()` 和 `event()` **都没有实时调用 `publish_progress()` 上报训练epoch**。历史遥测系统不等于C7实时训练监控，这正是用户看不到进度的直接软件设计原因之一。
- 旧C6日志增强版有66个完整`EPOCH_END`事件，成功结束后才从Kaggle output API取到；不能将C6较短用时直接等同于C7：C7共100个epoch，严格确定性设置可能改变运行速度。没有C7 epoch轨迹前，不能预计准确完成时间、报出训练百分比或准确率。

## 原因归类

**已证实：监控/可观测性缺陷。** Kaggle运行期间CLI没有返回日志或输出文件；应用层进度只写在Kaggle容器本地，GitHub监控读取不到。这解释了GitHub页面长期只显示 `RUNNING` 而看不到 `EPOCH`。

**尚未证实：是否GPU算子卡住、是否数据加载死锁、是否Kaggle平台阻塞。** 在当前已有外部证据里不能判断。只有 `COMPLETE` 后读取训练日志、`ERROR` 错误、或Kaggle Notebook UI真实单元进度才可进一步诊断。对未能访问的进度不可虚构。

## 立即规避重复费用与数据损失

1. **不要重启仍显示RUNNING的同一内核、不要覆盖当前 Notebook 代码。** Kaggle Kernel Push 会产生新的训练版本或重新启动任务。
2. 当前 GitHub 长运行监控工作流最多等待440×40秒（约4小时53分钟，外加配置开销），即使只在等待Kaggle，也会占用GitHub托管Runner分钟、私有仓库额度；这并不意味着GitHub执行了模型GPU计算。
3. 已新增独立按需工作流 `kaggle-c7-collect-on-demand.yml`：只在远程内核 `COMPLETE` 后收集 `results.json`、4份25轮 `history.json`、4份 `best.pt`、完整 `live-progress.jsonl` 并验收归档，**不启动或重启GPU**。用来兜底当前长轮询超时造成的归档遗漏；它仅支持手动调用，不在每次主分支提交时运行。
4. 今后改为：GitHub小作业提交Kaggle并立即退出；训练在Kaggle执行；独立定时/按需状态检查不占用几个小时的Actions Runner；GPU结束后再开一个短作业收集。确实需要训练中实时epoch时，须在新Notebook里采用合适的独立进度上报（含网络连通性与隐私评估），或将4组任务拆分为较短独立版本；当前运行版本无法事后注入心跳代码。

## 安全与发布边界

未修改现有Kaggle GPU任务，未上传新手机人脸样本，未更改冻结TEST/模型权重，未触发训练重启。**C7目前的准确率、是否过门槛、FPGA可部署性都尚无证据。**
