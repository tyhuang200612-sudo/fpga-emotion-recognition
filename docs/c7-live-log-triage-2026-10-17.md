# C7 GPU 实时日志不可见调查（2026-10-17）

## 结论与确认事实

1. C7 GitHub Actions [#39010774856](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39010774856) 已成功提交一个独立 Kaggle 私有 Notebook `tyhuanga/fpga-c7-gap2-fixed25-deterministic`，远程Notebook与预期训练代码、冻结教师数据集挂载的元数据校验通过。GitHub 的运行状态停在“Track C7 fixed25 GPU progression and retrieve available logs”，说明 GitHub **在等待 Kaggle**，不是证明GPU算到了哪一轮。
2. 两次互不修改GPU的 GitHub 只读检查：
   - [#39014326452](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014326452) 通过 Kaggle CLI 直接查询状态：`KernelWorkerStatus.RUNNING`；`kaggle kernels files` 返回空列表、`kaggle kernels logs` 无有效输出、尝试只读取 `live-progress.jsonl` 未返回文件。Kaggle pull 的真实 Notebook **确实包含** `run_c7()`、固定25 epoch、每轮 `c6_progress("EPOCH_END"` 等执行与进度代码；代码存在不代表正在执行。
   - [#39014368690](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39014368690) 再次从 Kaggle CLI 直接查询 RUNNING，日志只有一个换行字节；`kaggle kernels files` 列表为空，筛选 `live-progress.jsonl`/`progress.jsonl`/`results.json` 的只读输出 API 无任何下载文件。无可独立复核的训练 Epoch/Batch 进度。
3. **重要时间解释**：先前 C6 真实归档 `live-progress.jsonl` 在 2026-10-17T06:03:08Z 进入 START，在T06:20:12Z 完成，期间共66个 Epoch，按每轮约13～15秒估算 C7 100轮单纯 GPU 训练约23分钟（严格CUDA确定性算法可能更慢）。但 C6 的 GitHub Actions 任务从T05:22:33Z启动，至T06:03:08Z才有GPU代码 START，中间约 **40分钟为Kaggle启动、排队、环境准备或其他前置等待**。不能把 GitHub Action 的累计等待时间当GPU纯训练时间、据此认定C7卡死。
4. **日志根因（可证实部分）**：训练代码确实每25个Batch和每个Epoch既 `print(...,flush=True)`，又将JSONL写入 `/kaggle/working/.../live-progress.jsonl`。但Kaggle公开的运行中CLI日志和输出API当前均不返回这些内容。这里存在**日志平台可见性限制**；不可据此断言Python未执行，亦不可据此断言模型训练正常。
5. **风险和未知**：尚未验证 C7 实际epoch计数、GPU占用率、内核是否一直停在同一算子。严格确定性cuDNN、cuBLAS可能造成比C6更慢的执行路径；只有Kaggle完成状态及真正的 `results.json`、四组25轮 `history.json` 与100条 `EPOCH_END` 能可靠证明完整训练。
6. **没有重启、取消或覆盖任何 Kaggle C7 训练；没有导出修改模型到Android或FPGA**。

## 已执行的操作
- 通过 GitHub Actions 独立只读诊断，不修改任何 Kaggle Notebook、作业状态或 GPU 设置，排除“只看GitHub RUNNING”的伪证据。
- 检查实际远程 Notebook 内容与关键训练/日志代码相符；比较真实历史 C6 JSONL 时间戳，修正可能误导的训练耗时估计。
- GitHub 现有监控任务在一个长时间的托管runner shell里每40秒轮询 Kaggle 状态，这**占用 GitHub Actions 的计费/免费配额**，并且由于Kaggle API mid-run 输出为空，几乎不提供训练健康度的新证据。今后的训练应拆分为“很短的GPU提交作业”“用户选择的低频短任务检查”“成功后短任务下载结果”，不要让 GitHub Runner 连续等待数小时。

## 后续处置原则
- 此刻保留 C7 单次正在运行的任务，避免正在训练时丢掉100轮成果。不要因为RUNNING日志空白而循环强制重跑。
- 如 Kaggle `COMPLETE`，逐条检验 `results.json`、四份`best.pt`、四份`history.json`每组25epoch、100个`EPOCH_END`，再比较CE/KD准确率、类召回、误判和稳健升级门槛。
- 如 Kaggle `ERROR`，读取最终的 Kaggle日志、traceback，分析是否CUDA确定性算子报错、GPU OOM、数据加载异常，再决定针对性修复。不应直接取消并原样重训。
- 如明显超出C6排队+训练耗时数倍且仍无结果，优先由Kaggle UI直接查看“Active session”的GPU监控/Notebook执行格；如用户明确同意停止，再安全退出并采用真正分阶段持久化的训练设计。无法仅凭CLI外部RUNNING获取内存占用或GPU百分比。
