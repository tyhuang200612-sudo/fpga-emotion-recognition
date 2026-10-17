# C6 — 在 C5 GAP2 网络上隔离验证 C3 强教师蒸馏（2026-10-17）

**状态以 [C6 GitHub Actions #39006028260](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions/runs/39006028260) 和 [私有 Kaggle Notebook](https://www.kaggle.com/code/tyhuanga/fpga-c6-gap2-frozen-c3-kd) 为准。** 本文件创建时尚无正式验证成绩。不能因工作流成功提交就声称性能提高。

## 必要背景
- 已有历史最佳小学生 C2-C FER2013 验证 Macro-F1=0.751586、neutral/sad互相误判408；但在同一短时间连续采集的19张被用户确认是悲伤的手机裁剪图上，C2-C 0/19、C3 ResNet18 强教师18/19。它们非独立样本，不代表正式手机准确率。
- C5 GAP2 在相同训练条件下比 GAP1 平均 Macro-F1 高0.006464，虽有希望，但验证平均0.750498仍未超过C2-C，两个种子均未通过发布资格。
- 因此 C6 只测试单一新因素：保持 C5 的2×2学生结构、真实冻结C2026原权重初始化、2种子训练设置与常规增强不变，**仅将普通交叉熵CE损失变成80% CE+20%固定强教师KL蒸馏（温度T=2）**；不增加C4中复杂门控、pairwise、margin或新数据增强，避免同时改多个变量。

## 试验控制
| 设置 | 组A | 组B |
|---|---|---|
| 名称 | `GAP2_CE_control` | `GAP2_C3_KD` |
| 模型 | TinyCNN-C 16/32/64/64 + GAP2 FC256→3 | 完全相同 |
| 初始化 | C2026原始冻结权重，FC权重按4象限除以4 | 完全相同 |
| 随机种子 | 20261021、20261022 | 完全相同 |
| 优化器 | AdamW lr2e-4, weight_decay0.01 | 完全相同 |
| 训练次数 | 最多25 epoch，早停7，验证Macro-F1选权重 | 完全相同 |
| 损失 | 加权 CE | 0.80 加权 CE +0.20 KL，T=2 |
| C3 ImageNet ResNet18 教师 | 无 | 训练时冻结、eval、requires_grad=False |
| 训练/验证图片 | 原始 FER2013 干净拆分 13571 / 2979 | 完全相同 |
| FPGA/Android部署 | 无 | 无 |

教师checkpoint SHA256：`ce455ee8060e19d87be16d50926c881b672d172c2ea260bf0ad5e01572861b0c`；云端必须先验证SHA及教师原VAL表现。实验保留教师仅在GPU训练时推理，不会将教师部署到安路FPGA。

## 严格的独立验证与验收
- 原 FER2013 TEST **不参与本轮训练、选权重和调阈值**；原19张短时连续手机样本不上传、不加入训练，也不用它们选出所谓最优学生。
- 每个随机种子的 Macro-F1 必须比同期CE控制组**和**历史C2-C都高至少0.005，且中性↔悲伤互相误判数至少比两者少10；中性和悲伤召回相对C2-C各不允许恶化超过0.02，才算双种子稳健通过。
- 必须报告各自验证 Macro-F1、accuracy、中性与悲伤 recall、双向混淆矩阵、误判数与最佳epoch；不能只展示最好单次实验。
- 即使通过FER验证，**下一步也只是独立手机摄像头评估**。应获得采集者知情同意，并按人和拍摄时段分组隔离；不能用重复连续视频帧误当独立样本。
- FPGA资源目前仅按2×2头理论增加576权重估算，**未验证INT8量化、RTL综合、BRAM、时序和真机**；不能自动更新上板模型。

## 代码
- `tools/train_c6_spatial_kd.py`：冻结教师加载、纯控制CE/蒸馏损失、双种子训练与发布门槛。
- `tools/build_kaggle_c6.py`：基于已验证的C5 notebook创建私有Kaggle T4任务。
- `tests/test_c6_spatial_kd.py`：两组初始模型一致、蒸馏数值检查、notebook测试集隔离。
- `.github/workflows/kaggle-c6-spatial-kd.yml`：GitHub检查、Kaggle上传、监控、下载全部4份`best.pt`及`results.json`、归档。
- 归档产物预期名：`TinyCNN-C6-GAP2-CE-vs-C3Teacher-KD-TwoSeeds-ValidationOnly`，保留30天，须在交接时下载并校验SHA256。
