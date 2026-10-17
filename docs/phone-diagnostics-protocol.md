# C2-C 手机误判诊断（第一阶段，不训练 C5）

当前 APK 构建：[GitHub Actions #39003308849](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39003308849)

## 严格边界
- 使用 C2-C `C_distilled/best.pt`，seed=2026，epoch=12，结构 16/32/64/64，48×48×1；不替换原版 C、C2-C、FPGA INT8 ROM。
- 独立安卓包名：`com.tyhuanga.fpga.camerac2cdiag`。
- 本机 FP32 ONNX + 摄像头 ML Kit；不是 FPGA INT8 实测，也不代表心理情绪检测。
- 默认仅内存处理；没有网络权限、不自动存人脸图片。**只有选择真实表情并点击保存才写入**APP私有文件目录。只有点击“手动导出”并通过系统选择文件位置，才产生可共享的 ZIP。
- 默认不收集身份、账号、姓名；导出的 48×48 图仍属于敏感的人脸图像。务必本人自愿或获得明确同意，严禁未经许可上传。
- 删除 APP 内样本会清空原始保存数据，但不能删除已经手动导出的副本。

## 新版诊断能力
1. **裁剪比例**：1.00×、1.15×、1.30×、1.36×（旧默认）、1.40×。仅修改人脸框外扩比例，权重不变。
2. **原始与 EMA 平滑对照**：显示每帧真实模型原始概率最大值、平滑最大值及全类别平滑概率。
3. **实际模型输入预览**：渲染真正输入网络的 48×48 归一化灰度图，不是未裁剪的摄像头预览。
4. **消失/停止帧重置**：无脸后清空历史 EMA、输入图和输出，避免下一张脸沿用上一帧结果；ML Kit 单次完成事件处理/统一关闭 Image，避免异步关闭竞态。
5. **用户标注数据**：每次只手动保存一张 PNG 48×48 和 JSON，包含真实表情、裁剪比例、原始与平滑概率和时间戳；可导出 ZIP，或确认后删除。

## 建议操作（样本诊断，不是正式准确率基准）
- 在明亮、逆光、普通室内等环境测试；测试时保持手机位置和表情尽量稳定。
- 对每个裁剪比例，分别尝试中性、开心、悲伤。每种表情/比例各挑几次典型正确与错误结果。
- 如果不同裁剪比例下正确率明显变化，优先检查人脸裁剪/预处理。
- 如果每帧**原始预测**正确但平滑预测错误或迟滞，优先检查时序/EMA。
- 如果原始与平滑一致地错误，而灰度图清晰居中，问题更可能在模型/标签/域适应。
- 不要用这些手动挑选的困难样本估算真实准确率：它们**不是随机抽样的独立验证集**。
- 后续要评估真实手机效果，需先制定经同意采集的、按人分组的独立测试集，避免把同一个人的相邻视频帧分在训练和测试里。

## 后续使用
- 完成误判定位后，才能决定需要的是手机预处理修复、数据复核还是重新训练 C5。
- 若需要云端比较 C3 教师与 C2-C 学生，在获知用户许可和隐私要求后，使用同一组48×48裁剪输入做离线批处理评测；不要自动上传图片至公有 GitHub。
- APK 编译完成且校验后，提交实际包的 GitHub Actions run 与 SHA256 到交接索引。

## 已完成诊断包（2026-10-17）
- 最终 GitHub Actions：<https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39003424873>，全部步骤通过。
- 完整的 APK 工件：`TinyCNN-C2C-Phone-Diagnostic-arm64-APK`，artifact ID **12089622870**；下载包可与旧版 C、C2-C 并存。
- 安卓应用 ID：`com.tyhuanga.fpga.camerac2cdiag`。
- 本轮实发 APK：`TinyCNN-C2C-Phone-Diagnostic-arm64.apk`（33.49 MiB，SHA256：`3da64020c7a7542c7e25e5647aee666e117f2067cd80a1e6fc3ab7b9d15fc531`）。
- APK 的 `assets/model-info.json` 校验：`variant=C_distilled`、`selected_epoch=12`、`model=TinyCNN-C2-C Distilled Diagnostic (unchanged weights)`、`manual_samples_only=true`、`no_network_permission=true`；ONNX/PyTorch 输出偏差小于 1e-4。
- APK 仅完成 CI 编译和产物静态校验，**尚未在用户实际手机上完成运行体验和真实标签诊断**。
