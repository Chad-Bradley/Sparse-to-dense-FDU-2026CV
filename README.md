# s2d_baseline

## 目录结构

```
s2d_baseline/
├── DESIGN.md                 设计、指标口径、里程碑、待定决策
├── FORMULAS.md               损失与指标的完整公式
├── train.sh                  基础训练启动脚本（默认 upproj + rgbd + 500）
├── README.md                 本文件
├── s2d/
│   ├── config.py             配置 dataclass
│   ├── utils.py              种子 / checkpoint / JSON
│   ├── viz.py                四联图与误差图
│   ├── analysis.py           边界 / 无纹理分桶
│   ├── losses.py             MaskedL1 / MaskedMSE
│   ├── metrics.py            image / pixel 两种口径
│   ├── engine.py             train / evaluate 循环
│   ├── data/
│   │   ├── dataset.py        NYU h5 Dataset
│   │   ├── transforms.py     几何增广 + 颜色抖动
│   │   └── sparse.py         严格 N 点稀疏采样
│   └── models/
│       ├── encoder.py        ResNet50（4 通道 conv1）
│       ├── decoder.py        deconv2 / deconv3 / upconv / upproj
│       └── sparse2dense.py   完整网络
├── scripts/
│   ├── train.py              训练入口
│   ├── evaluate.py           评估 + 四联图
│   ├── analyze.py            边界 / 纹理分桶
│   └── debug_sample.py       单样本数据自检
└── tests/                    单元测试
```
## 快速开始

```bash
cd s2d_baseline
bash train.sh
```

`train.sh` 会自动完成：

1. 定位并激活 conda（默认 `CONDA_ENV=s2d`）；
2. 设置 `CUDA_VISIBLE_DEVICES`；
3. 设置 `TORCH_HOME=/root/Test/hf_cache`；
4. 设置 `S2D_RUNS=/root/Test/s2d_baseline_runs`；
5. 检查数据目录和 CUDA；
6. 把日志 tee 到 `$S2D_RUNS/logs/baseline_*.log`。

训练输出默认在 `$S2D_RUNS`；直接 `python -m scripts.train` 时默认在
`s2d_baseline/runs/`（根盘紧张，建议始终用 `train.sh` 或显式设 `S2D_RUNS`）。

## `train.sh` 全部环境变量

| 变量 | 默认值 | 可选值 / 说明 |
|---|---|---|
| `GPU` | `0` | 物理卡号；未设置时沿用已有的 `CUDA_VISIBLE_DEVICES` |
| `CONDA_ENV` | `s2d` | conda 环境名 |
| `ARCH` | `resnet50` | `resnet18` / `resnet50` |
| `DECODER` | `upproj` | `deconv2` / `deconv3` / `upconv` / `upproj` |
| `MODALITY` | `rgbd` | `rgb` / `d` / `rgbd` |
| `SAMPLES` | `500` | 稀疏点数；`rgb` 模态下无效 |
| `EPOCHS` | `15` | 总 epoch 数 |
| `BATCH_SIZE` | `8` | 训练 batch |
| `LR` | `0.01` | 初始学习率（每 5 epoch ×0.1） |
| `WORKERS` | `10` | DataLoader worker 数 |
| `CRITERION` | `l1` | `l1` / `l2` / `mse` |
| `SEED` | `0` | 随机种子 |
| `PRETRAIN` | `1` | `1` 加载 ImageNet 权重；`0` 等价 `--no-pretrain` |
| `METRIC` | `image` | `image`（原版口径）/ `pixel`（按像素累计） |
| `DATA` | `nyudepthv2` | 数据集名，对应 `<repo>/data/<DATA>` |
| `DATA_ROOT` | `${S2D_DATA_ROOT}` 或 `<repo>/data/<DATA>` | 优先级最高，可直接指定绝对路径 |
| `S2D_DATA_ROOT` | 空 | 数据根目录的环境变量写法；`Config` 也会读取它 |
| `TORCH_HOME` | `/root/Test/hf_cache` | 预训练权重缓存 |
| `S2D_RUNS` | `/root/Test/s2d_baseline_runs` | 训练输出根目录 |
| `RUN_NAME` | 空 | 空 = 自动拼名；非空时作为输出子目录 |
| `RESUME` | 空 | checkpoint 路径；续训时用 |
| `LIMIT_TRAIN` | `0` | 只用前 N 张训练图（0 = 全量，试跑用） |
| `LIMIT_VAL` | `0` | 只用前 N 张验证图（0 = 全量，试跑用） |
| `PRINT_FREQ` | `50` | 每多少个 iter 打印一次 |
| `LOG_DIR` | `$S2D_RUNS/logs` | 日志目录 |

额外参数会原样追加到 Python 命令末尾：

```bash
bash train.sh --epochs 2 --workers 4
```

## `scripts/train.py` 全部命令行参数

`train.sh` 已经封装了大部分；直接调用 Python 时可用下面这些。

### 数据

| 参数 | 默认 | 可选值 / 说明 |
|---|---|---|
| `--data-root` | `$S2D_DATA_ROOT` 或 `<repo>/data/nyudepthv2` | h5 根目录 |
| `-m, --modality` | `rgbd` | `rgb` / `d` / `rgbd` |
| `-s, --num-samples` | `500` | **严格无放回**采样 N 个点，不是概率采样 |
| `--max-depth` | `-1.0` | 负数 = 不限制最大深度 |
| `--sparsifier` | `uniform` | 目前只有 `uniform` |
| `--no-rgb-normalize` | 关闭 | 加了这个参数才关闭 ImageNet 归一化 |
| `--lighting-std` | `0.1` | Lighting 强度；`0` = 关闭 |

### 模型

| 参数 | 默认 | 可选值 / 说明 |
|---|---|---|
| `-a, --arch` | `resnet50` | `resnet18` / `resnet50` |
| `-d, --decoder` | `upproj` | `deconv2` / `deconv3` / `upconv` / `upproj` |
| `--conv1-init` | `random` | `random`（与上游一致）/ `copy`（前三通道拷 ImageNet，第 4 通道置 0） |
| `--bottleneck-kernel` | `1` | 瓶颈卷积核；上游用 1，可试 3 |
| `--no-pretrain` | 关闭 | 不加载 ImageNet 权重 |

### 优化与指标

| 参数 | 默认 | 可选值 / 说明 |
|---|---|---|
| `--epochs` | `15` | 总 epoch 数 |
| `-b, --batch-size` | `8` | 训练 batch |
| `--eval-batch-size` | `1` | 验证 batch；image 口径要求 1 |
| `--lr` | `0.01` | 初始学习率 |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `1e-4` | weight decay |
| `--lr-step` | `5` | 每多少个 epoch 衰减一次 |
| `--lr-gamma` | `0.1` | 每次衰减乘数 |
| `-c, --criterion` | `l1` | `l1` / `l2` / `mse` |
| `--metric-reduction` | `image` | `image` / `pixel` |
| `--workers` | `10` | DataLoader worker 数 |
| `--seed` | `0` | 随机种子 |

### 输出与试跑

| 参数 | 默认 | 说明 |
|---|---|---|
| `--out-dir` | `s2d_baseline/runs`（或 `$S2D_RUNS`） | 输出根目录 |
| `--run-name` | 空 | 空 = 自动拼名 |
| `--print-freq` | `50` | 打印间隔 |
| `--save-samples` | `8` | 目前训练脚本只解析，未使用；可视化由 `evaluate.py` 负责 |
| `--limit-train` | `0` | 只用前 N 张训练图（试跑） |
| `--limit-val` | `0` | 只用前 N 张验证图（试跑） |
| `--resume` | 空 | checkpoint 路径 |

> `--limit-train` / `--limit-val` 不写进 config，只用于试跑。

## 解码器选项

| 名称 | 每级操作 | 参数量（整体，ResNet50） |
|---|---|---|
| `deconv2` | ConvTranspose2d(k=2, s=2) → BN → ReLU | 约 28M |
| `deconv3` | ConvTranspose2d(k=3, s=2, p=1, op=1) → BN → ReLU | 约 32M |
| `upconv` | Unpool → 5×5 conv → BN → ReLU | 约 43M |
| `upproj`（默认） | Unpool → 双分支 5×5/3×3 → 相加 → ReLU | 约 64M |

四种都是 4 级、每级分辨率 ×2、通道 ÷2。

## 输出目录

`<out_dir>/<run_name>/`，其中 `run_name` 默认是：

```
{arch}_{decoder}_{modality}_s{num_samples}_{criterion}
```

目录已存在时会追加时间戳，不覆盖旧实验。

| 文件 | 内容 |
|---|---|
| `config.json` | 训练开始时的配置快照 |
| `metrics.csv` | 每 epoch 一行：`epoch,lr,train_loss,train_rmse,train_absrel,val_rmse,val_absrel,val_mae,val_lg10,val_delta1,val_delta2,val_delta3` |
| `last.pth.tar` | 最新 checkpoint（每 epoch 覆盖） |
| `best.pth.tar` | 验证 RMSE 最小的 checkpoint |

说明：这个 commit 的版本**没有** `best.txt`、进度条、`epoch_time`/预计总时长；
那些是后来在 `s2d_repro` 里加的。best 选择标准是**验证集 image RMSE 最小**。

## 评估、分析与自检

```bash
# 指标 + 四联图 + 误差图（默认写到 checkpoint 目录的 eval/）
python -m scripts.evaluate --checkpoint runs/<run>/best.pth.tar --save-samples 8

# 边界 / 无纹理分桶（默认写到 checkpoint 目录的 analysis/）
python -m scripts.analyze --checkpoint runs/<run>/best.pth.tar --bins 4 --stride 8

# 单样本数据自检
python -m scripts.debug_sample --split val --index 0 --modality rgbd --num-samples 500
```

`evaluate.py` 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` | 必填 | `best.pth.tar` 路径 |
| `--split` | `val` | `train` / `val` |
| `--data-root` | checkpoint 里的配置 | 数据根目录 |
| `--output-dir` | `<checkpoint 目录>/eval` | 输出目录 |
| `--save-samples` | `8` | 保存多少张四联图，0 = 不保存 |
| `--workers` | `4` | DataLoader worker 数 |
| `--device` | `cuda` | 计算设备 |

`analyze.py` 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` | 必填 | `best.pth.tar` |
| `--split` | `val` | `train` / `val` |
| `--data-root` | checkpoint 里的配置 | 数据根目录 |
| `--output-dir` | `<checkpoint 目录>/analysis` | 输出目录 |
| `--bins` | `4` | 分桶数 |
| `--stride` | `8` | 每 N 个有效像素取 1 个；1 = 全量 |
| `--workers` | `4` | DataLoader worker 数 |
| `--device` | `cuda` | 计算设备 |

## 单元测试

```bash
python -m unittest discover -s tests -v
```

## 常用示例

```bash
# 默认：upproj + rgbd + 500
bash train.sh

# 指定 GPU
GPU=3 bash train.sh

# 快速试跑（几十秒）
EPOCHS=1 WORKERS=0 LIMIT_TRAIN=64 LIMIT_VAL=64 bash train.sh

# 换解码器 / 改 batch
DECODER=deconv3 bash train.sh
BATCH_SIZE=4 bash train.sh

# 200 点对照
SAMPLES=200 bash train.sh

# 不加载 ImageNet 权重
PRETRAIN=0 bash train.sh

# 续训
RESUME=/root/Test/s2d_baseline_runs/<run>/last.pth.tar bash train.sh
```
