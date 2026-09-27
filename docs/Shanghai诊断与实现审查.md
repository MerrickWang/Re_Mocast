# Shanghai 诊断与实现审查

本次以现有验证集选择和诊断设置；保留测试集用于最终评估。
这些工具不会通过 `prepare_run` 改写训练目录里的 config/checkpoint。

## 1. 持久性基线与逐预测步结果

在服务器项目根目录运行：

```bash
python -u -m mocast.tools.diagnose_forecast -c outputs/shanghai_5090/config.yaml --checkpoint outputs/shanghai_5090/checkpoints/best.pt --device cuda --out outputs/diagnostics/forecast
```

默认验证集、FP32，与原来的独立测试评估精度一致；加 `--amp` 使用保存配置中的混合精度。
如需比较 AMP，另用 `--out outputs/diagnostics/forecast_amp` 保留两份报告。
默认完整验证集；`--max-batches 2` 仅用于快速检查，报告会标注 `partial`。
时间间隔从数据上下文/配置读取（Shanghai 名义间隔 6 分钟），也可通过
`--interval-minutes` 指定；H5 未提供时间戳，不能从文件确认实际采样间隔。

输出：

- `forecast_diagnostics.json`：每个预测步、每个阈值的命中/漏报/误报、CSI、HSS、precision、recall、覆盖率。
- `lead_metrics.csv`：便于导入表格，包含 `csi_macro` 和逐阈值结果。
- `lead_skill.png`：第 1～20 步平均 CSI 和物理单位 MSE 曲线。

四个分支都使用同一批验证样本和相同阈值：

| 分支 | 含义 |
|---|---|
| model | 原始完整模型输出，没有乘校准系数 |
| persistence | 重复最后一帧作为所有未来预测 |
| motion_only | 保留已学习运动场，关闭未来源汇增量 |
| source_only | 关闭未来平流，只累加已学习源汇增量 |

后两项是现有权重的贡献探测，不是重新训练的论文消融。
`first_lead_below_persistence` 是首次低于持久性基线的预测步；同时报告全部低于基线的步，不能把首次交叉理解为之后永远更差。
源汇和运动幅值逐步统计保存在 `branch_stats`，列解释在 `branch_stat_columns`。
幅值的单位遵循模型配置；源汇为归一化值，不能直接标为 dBZ。

## 2. 运动、平流、源汇和 Eq.13 审查

依据本地 `23353-AAAI26.WuB-DM.pdf` 第 5 页 Eq.12/13 及实验设置。

确认和修正：

1. **BF16 采样网格精度**：原实现用运动张量的 BF16 dtype 构造坐标网格，再由 grid_sample 自动转换为 FP32，已丢失的坐标精度不能恢复。
   128² 方块、零运动、20 步实验，旧实现 MSE 约 0.003048，修复后该实验为 0；修复为在生成坐标前就转 FP32，并以 FP32 采样。
   模型参数形状不变，旧权重仍可加载。该修复会影响 AMP 路径，不能假定它足以恢复论文分数。
2. **运动损失重复平均**：Eq.13 是对运动差的平方向量范数加权求和，再除以掩码总量。
   旧代码除以掩码总量后，还除以 2 个通道并对时间步取平均。5 帧输入时相对该公式缩小 6 倍。
   已改为每个样本加权范数总和 / 掩码总量，最后对 batch 取平均，并用 FP32 累加。
3. **掩码顺序**：改为先分别平均连续帧、再取 max 和阈值；随后下采样保留平均覆盖率。
   原 `any/majority` 保留为显式兼容选项，论文对应默认值为 `average`。
4. **相邻运动梯度**：Eq.13 没有 stop-gradient，新的默认训练配置改为 `detach_target=false`。
5. **逐时刻 CSI 口径**：旧实现将多个阈值的混淆计数混合后求 CSI；已与总分统一为各阈值独立计算再取平均。
   总体 CSI 计算本身未更改，因此该修复不会直接抬高此前 0.2028 的总分。
6. **图像时间标签**：去掉硬编码的每帧 5 分钟，按数据配置/上下文生成。

已验证但未认定为错误：

- `(dx,dy)` 顺序与正 dx 向右移动的方向一致；正向内容位移用反向采样 `p-flow` 实现。
- 像素单位和归一化单位在明确换算后给出相同平移。
- 重建为上一预测帧平流后加源汇；恒定正源汇按时间累加。
- 多次双线性亚像素平流本身会削弱尖锐峰值。报告中 `operator_probes` 对比重复 0.5 像素移动 20 次与单次移动 10 像素。
  单像素脉冲实验中重复插值峰值约 0.1762，单次平移约 1.0。这个常量场实验不能作为直接替换可变运动场重建的依据，暂不擅改论文递推公式。

仍未能对齐作者实现的内容：PMM/时序网络的具体宽度、潜空间运动尺度、注意力缩放、优化器细节和超参搜索结果。
它们仍需作者源码/配置比对；不能将本次修复声明为完成数值复现。

## 3. H5 编码、预处理与划分审计

```bash
python -u -m mocast.tools.audit_h5 -c outputs/shanghai_5090/config.yaml --out outputs/diagnostics/h5_audit.json
```

默认每个实际 split 固定随机抽取 32 个序列，每序列抽 5 帧做像素统计，整个抽中序列用于 SHA256 检查。
全量内容检查（较慢）：

```bash
python -u -m mocast.tools.audit_h5 -c outputs/shanghai_5090/config.yaml --samples-per-split 0 --frames-per-sequence 25 --out outputs/diagnostics/h5_audit_full.json
```

报告包含组数量、形状、dtype、属性、实际划分、编号重叠、抽中序列的精确重复、非空帧的跨 split 精确重复。
没有发现重复不能证明事件独立（尤其是抽样检查）；没有时间戳就不能确认窗口是否重叠或来自同场降水。

同时统计 `pixel_0_255` 与 `dbz` 两种候选解释的值域、越界量、阈值覆盖率，以及按当前 sanitize/crop/resize 处理后的覆盖率和平均帧峰值。
注意当前 sanitize 将越界值替换为 fill=0，不只是将强值截断为 70；直接把编码灰度当 dBZ 会大量清零。
两种解释仅用于审计，不自动修改训练配置。

本地真实文件抽样结果（seed=2026，每 split 32 个序列）：
train 原值范围 0～236，val 0～209，test 0～211，未发现抽中序列精确重复或抽中非空帧跨 split 重复。
在 `pixel_0_255` 假设下，验证集抽样 40 dBZ 覆盖率由缩放前 0.1229% 降至 0.1173%，约下降 4.5%。
该量级不能单独解释此前模型 40 dBZ 召回率仅 5.3%；这只是抽样结论，且依赖编码假设。
根/组属性没有提供编码来源。仍需数据制作方提供：

- PNG 到 H5 的生成脚本或原始说明；灰度到 dBZ 公式，缺测编码。
- 裁剪区域/缩放方式；帧间隔；是否已去除弱回波。
- 原始事件或时间索引，以及论文使用的 train/val/test 划分。

## 修复后重训

先运行诊断和审计。确认数据编码后，新训练应另用输出目录并从头开始。
保留原有物理设置、数据路径和划分以便对照；保存的旧 config 不会自动继承新默认值，因此需要显式覆盖损失选项：

```bash
python -u -m mocast.tools.train -c outputs/shanghai_5090/config.yaml --device cuda --output outputs/shanghai_5090_fixed --set run.resume=null loss.motion.reduce=average loss.motion.detach_target=false
```

不要恢复旧优化器继续跑并把结果当作从头训练的可比实验。新结果先按同一验证集判断，再进行最终测试集评估。

## 本次文件清单与验证

新增：`mocast/tools/diagnose_forecast.py`、`mocast/tools/audit_h5.py`、
`mocast/tests/test_diagnostics.py`、本文档。

修改：`mocast/models/advection.py`、`mocast/losses/__init__.py`、
`mocast/configs/train/default.yaml`、`mocast/metrics/csi.py`、
`mocast/tools/visualize.py`、`mocast/tests/test_advection.py`、
`mocast/tests/test_losses.py`、`docs/作者确认清单.md`、`README.md`。

回归验证：57 项测试通过（diagnostics、advection、losses、model_contract、config）；
此前首轮含 metrics 的测试也通过。诊断工具已使用小模型权重完成端到端输出测试。
真实 H5 审计报告位于 `outputs/diagnostics/local_h5_audit.json`。
当前本地没有服务器训练的 best.pt，所以未运行该权重的完整验证集预测诊断，也未验证 5090 的新训练分数。
