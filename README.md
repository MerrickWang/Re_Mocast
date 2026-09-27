# Re_Mocast · MoCast 非官方复现

基于论文 *MoCast: Learning Turbulent Motions Under Physical Guidance for Precipitation Nowcasting* 的 PyTorch 实现，包含物理引导运动建模（PMM）、运动引导源汇建模（MSM）、时序预测、可微平流重建及实验性的 MoCast+ 残差扩散模块。

论文：Binqing Wu 等，AAAI 2026，[DOI: 10.1609/aaai.v40i19.38628](https://doi.org/10.1609/aaai.v40i19.38628)。本仓库并非作者官方代码。

**当前状态：训练、评估和诊断流程已跑通，但尚未达到论文数值。** Shanghai H5 / RTX 5090 单种子测试 CSI 约为 0.207。公开材料未明确的设置采用可配置假设，数据编码和事件划分仍待核实。请先阅读 [已知问题](docs/KNOWN_ISSUES.md) 与 [实验记录](docs/EXPERIMENTS.md)，不要将工程测试通过等同于完成论文复现。

## 实验结果

以下为维护者在 Linux / RTX 5090 上提供的测试结果，seed=2026，测试组 526 个序列，未应用强度校准。原始服务器日志与权重未随仓库发布。

| 实验 | Batch | CSI ↑ | HSS ↑ | CSI-P4 ↑ | CSI-P16 ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 修复前 | 2 | 0.202794 | 0.305182 | 0.237275 | 0.316191 | 0.738393 | 0.283719 | 22.725889 |
| 修复后 | 2 | 0.206922 | 0.309606 | 0.235675 | 0.311705 | 0.749314 | 0.280544 | 21.339763 |
| 修复后 | 6 | 0.206699 | 0.316441 | 0.255153 | 0.349098 | 0.662494 | 0.279979 | 23.594733 |

修复版包含 FP32 平流采样网格、运动损失归约/掩码修正等。BS=6 的池化 CSI 较高，但总体 CSI 基本不变，SSIM/MSE 更差，不能据此断定某个配置全面更优。结果为单种子观察，尚无多种子置信区间；感知指标和 MSE 还存在末批次等权汇总问题，见 [K05](docs/KNOWN_ISSUES.md#k05-评估批次权重与精度口径)。

## Linux / RTX 5090 安装

下列命令在仓库根目录执行。Python 建议使用 3.10 或更高版本。RTX 5090 需要支持 Blackwell 的 PyTorch 构建；[PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)说明了 CUDA 12.8 / Blackwell 支持。以下使用官方配套的 PyTorch 2.10.0 / torchvision 0.25.0 / cu128 作为安装示例，**不代表历史服务器实验已锁定到这套版本**。其他配套版本见 [官方版本列表](https://pytorch.org/get-started/previous-versions/)。

```bash
git clone https://github.com/MerrickWang/Re_Mocast.git
cd Re_Mocast

# 只使用 conda-forge，避免隐式访问 Anaconda defaults channels
conda create -n mocast5090 --override-channels -c conda-forge python=3.10 -y
conda activate mocast5090

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

nvidia-smi
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA 不可用：检查 NVIDIA 驱动和 PyTorch 构建'
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
x = torch.randn(32, 32, device='cuda')
print('CUDA matmul:', (x @ x).shape)
PY
```

没有 Conda 时，可先用 `python3.10 -m venv .venv` 和 `source .venv/bin/activate` 创建环境，再执行 pip 命令。仓库暂未提供全部依赖的锁文件；正式实验应保存 `pip freeze` 和 `nvidia-smi` 输出。

无真实数据的快速检查：

```bash
python -m mocast.tools.smoke --device cuda --batch 2
python -m mocast.tools.train \
  -c mocast/configs/experiments/synthetic_dryrun.yaml \
  --device cuda --output outputs/synthetic_check
```

## 准备 Shanghai H5

数据集需自行准备，不包含在仓库内。当前适配器支持的分组结构：

```text
Mocast_shanghai.h5
├── train/
│   ├── 0        uint8 [25, 501, 501]
│   ├── 1        uint8 [25, 501, 501]
│   ├── ...
│   └── all_len  标量，读取时忽略
└── test/
    ├── 0        uint8 [25, 501, 501]
    ├── ...
    └── all_len  标量，读取时忽略
```

每个序列前 5 帧为输入、后 20 帧为目标，完整空间范围双线性缩放至 128×128。当前文件 train 组 1534 个序列，以固定 seed=2026 留出约 10% 为验证集，实际为 **1381 train / 153 val / 526 test**；test 组保持原样。运行种子 `--seed` 与 `dataset.h5_split_seed` 分开管理，多种子实验应固定后者。

默认读取 `mocast/data/Mocast_shanghai.h5`，也可覆盖服务器上的绝对路径：

```bash
python -m mocast.tools.audit_h5 \
  -c mocast/configs/experiments/shanghai_h5_5090.yaml \
  --set dataset.h5_path=/path/to/Mocast_shanghai.h5 \
  --out outputs/diagnostics/h5_audit.json
```

配置暂按 `dBZ = pixel × 70 / 255` 解码，再按 [0,70] 归一化；**H5 中没有元数据确认这一编码**。仅当来源确认原值已经是 dBZ 时才使用 `dataset.h5_encoding=dbz`。当前越界值会被 sanitize 替换为 0，错误切换编码会改变大量回波。H5 未提供时间戳，不能证明事件独立；抽样重复检查为 0 也不能排除同场降水跨划分。

## 开始训练

以下使用已做过实验的 batch size 2，最多 200 epoch，以 `val_csi` 选择最佳检查点，patience=20。epoch 从 0 计数，早停可能使实际训练远少于 200 轮。

```bash
mkdir -p outputs/shanghai_5090_fixed

CUDA_VISIBLE_DEVICES=0 nohup python -u -m mocast.tools.train \
  -c mocast/configs/experiments/shanghai_h5_5090.yaml \
  --device cuda --seed 2026 \
  --output outputs/shanghai_5090_fixed \
  --set \
    dataset.h5_path=/path/to/Mocast_shanghai.h5 \
    run.resume=null \
    train.epochs=200 \
    train.batch_size=2 \
    train.num_workers=4 \
    train.amp=true \
    train.amp_dtype=bfloat16 \
    loss.motion.reduce=average \
    loss.motion.detach_target=false \
  > outputs/shanghai_5090_fixed/train.log 2>&1 &

echo "PID: $!"
```

将 `/path/to/Mocast_shanghai.h5` 替换为真实路径。`--set` 支持点号键覆盖 YAML。BS=6 对照实验可用 `mocast/configs/experiments/shanghai_h5_5090_bs6.yaml` 并指定新输出目录；使用上面命令时还需将显式覆盖的 `train.batch_size=2` 改为 6。batch size 改变每轮更新次数，应单独记录。

```bash
tail -f outputs/shanghai_5090_fixed/train.log
# 另一个终端
watch -n 2 nvidia-smi
```

`tail -f` 中的 Ctrl+C 只退出看日志，不会停止 nohup 训练进程。

产物包括 `config.yaml`、`run_info.json`、数据划分清单、`metrics.jsonl`、`summary.json`、`checkpoints/best.pt` 和 `checkpoints/last.pt`。保存的 config 是解析后的快照，修改源配置默认值不会更新它；使用旧快照重训应显式覆盖修复项。

当前 `run.resume` 可加载模型、优化器和 epoch，但不完整恢复调度器、随机状态与早停状态，暂不保证与无中断训练等价，见 [K06](docs/KNOWN_ISSUES.md#k06-断点恢复状态不完整)。上述对照实验应从头训练。

## 验证、测试与可视化

先检查验证集，诊断工具生成完整模型、持久性基线、关闭源汇和关闭平流的逐阈值/逐时效统计：

```bash
python -u -m mocast.tools.diagnose_forecast \
  -c outputs/shanghai_5090_fixed/config.yaml \
  --checkpoint outputs/shanghai_5090_fixed/checkpoints/best.pt \
  --device cuda --out outputs/diagnostics/forecast_fixed
```

输出为 `forecast_diagnostics.json`、`lead_metrics.csv`、`lead_skill.png`。默认 FP32，追加 `--amp` 可按保存的 AMP dtype 运行。`motion_only` / `source_only` 共用已训练的预测场，是事后贡献探测，并非重新训练的论文消融。

验证集确定配置后评估测试集；使用独立输出目录避免工具改写训练目录配置：

```bash
python -u -m mocast.tools.eval \
  -c outputs/shanghai_5090_fixed/config.yaml \
  --checkpoint outputs/shanghai_5090_fixed/checkpoints/best.pt \
  --split test --device cuda \
  --output outputs/shanghai_5090_fixed_eval \
  --set train.batch_size=2 eval.per_lead_time=true

python -u -m mocast.tools.visualize \
  -c outputs/shanghai_5090_fixed/config.yaml \
  --checkpoint outputs/shanghai_5090_fixed/checkpoints/best.pt \
  --split test --sample 0 --kind prediction --device cuda \
  --output outputs/shanghai_5090_fixed_visualize \
  --out outputs/shanghai_5090_fixed_visualize/visuals
```

测试输出为 `metrics_test.json`。CSI/HSS 对 [20,30,35,40] dBZ 分别累计混淆计数后求分数，再对阈值取平均；CSI-P4/P16 在最大池化后计算；MSE 在反归一化后的物理数值上计算。当前 Shanghai 图示使用名义 6 分钟间隔，H5 本身未提供时间戳。

首次 LPIPS 会下载 AlexNet 权重。下载缓慢时，可将完整的 `alexnet-owt-7be5be79.pth` 放入 `${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/`。快速检查可用 `eval.lpips=false`，此时不能报告完整 LPIPS。检查 JSON 的 `lpips_backend`，VGG 特征距离回退值不能当作官方 LPIPS。

强度校准系数只能在验证集选择并在测试前固定，旧模型系数不能自动沿用于新模型。目前没有正式校准 CLI，历史探索结果见实验记录。

## 目录与其他功能

```text
mocast/
├── configs/       数据、模型、训练、消融与搜索配置
├── datasets/      Shanghai H5/PNG、SEVIR、MeteoNet、合成数据
├── models/        PMM、MSM、时序网络、平流、MoCast/MoCast+
├── losses/        降水 MSE 与运动趋势一致性损失
├── metrics/       CSI/HSS/池化 CSI/SSIM/LPIPS
├── engine/        训练循环、验证、检查点
├── tools/         train/eval/visualize/audit_h5/diagnose_forecast 等
└── tests/         单元测试、集成测试、慢速过拟合检查
docs/
├── KNOWN_ISSUES.md
├── EXPERIMENTS.md
├── Shanghai诊断与实现审查.md
├── 作者确认清单.md
└── 复现验收报告.md
```

SEVIR/MeteoNet 配置、消融矩阵、超参搜索和 MoCast+ 入口仍保留，见 `mocast/configs/experiments/` 和 `python -m mocast.tools.<工具名> --help`。这些接口的存在不代表已在相应真实数据上完成数值验证；历史验收报告中的 L1/L2 主要是工程与合成数据测试。

## 测试与复现边界

```bash
python -m pytest mocast/tests -q -m "not slow"
python -m pytest mocast/tests -q
```

覆盖配置、数据读取、运动算子、损失、模型接口、指标和诊断工具。CPU 测试通过不能证明 RTX 5090 的端到端性能或论文指标一致。完整实验需记录依赖版本、Git commit、数据来源、划分、精度模式与种子。

重点问题：

- 强回波低估、长时预测变平滑，现有证据不能归因到唯一模块。
- 编码、裁剪规则、帧间隔和论文划分未核实。
- 部分网络宽度、注意力与优化器设置为推定默认值。
- 评估批次加权、精度一致性、断点恢复仍需完善。
- 服务器配置/权重未归档，多种子和修复后逐阈值对照不完整。

详情及后续优先级见 [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)。仓库仅包含代码、配置、测试和文档；数据、权重、输出、本地论文 PDF 和需求 DOCX 均排除。当前未附加开源许可证，代码与数据的使用/再分发许可需分别确认。
