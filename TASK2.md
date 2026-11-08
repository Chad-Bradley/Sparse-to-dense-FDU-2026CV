# 上采样与无效点处理的代码说明和运行方法

本次负责上采样消融和无效点处理，基于 GitHub 仓库提交 `21d5d940370b86c0632b0e4171fd69407ce91abd` 开发。与这个提交相比，当前版本新增 8 个文件，修改 8 个原文件。

## 一 新增了哪些文件

### 本次功能和配套文件

| 文件 | 用途 |
|---|---|
| `s2d/models/invalid_points.py` | 实现四种无效点处理方式，核心新增功能集中在这里 |
| `scripts/task2.py` | 生成、预览和启动本次分工的实验组合 |
| `tests/test_task2.py` | 检查前向、反向、权重加载，以及评估和分析入口 |
| `TASK2.md` | 本文，说明代码差异和使用方法 |

`invalid_points.py` 提供以下模式。

| 模式 | 具体处理 |
|---|---|
| `zero` | 缺测位置填 0，作为对照组 |
| `sentinel` | 缺测位置使用负数标记，默认填 `-1` |
| `mask` | 添加一个有效点通道，告诉网络哪里有测量值，RGBD 由 4 通道变成 5 通道 |
| `sparse` | 用两层带掩码卷积处理深度，再将深度特征和传播后的 mask 拼回 RGB |

`sparse` 采用深度输入前端的实现方式。两层卷积的通道变化为 `1→16→1`，第一层后接 ReLU。卷积只使用有效位置的输入，结果按邻域有效点数量归一化，同时将更新后的 mask 传给下一层。空邻域的输出为 0。

这一前端使用 PyTorch 稠密算子执行，后面的 ResNet 和解码器沿用已有结构。它能否改善精度、运行速度如何，需要后续实验验证。

`task2.py` 将三种解码器、两种最终输出头和四种无效点处理方式组合起来，共提供 24 组实验。默认运行时打印命令，添加 `--run` 后启动所选实验。

### 为运行入口补全的数据文件

GitHub 提交的 README、训练脚本和测试引用了 `s2d/data/`，但提交中缺少对应源码。因此，本次按仓库已有接口独立实现了下面四个文件。

| 文件 | 用途 |
|---|---|
| `s2d/data/__init__.py` | 定义数据模块包 |
| `s2d/data/dataset.py` | 读取 NYU HDF5，完成预处理、稀疏采样和 RGB/深度拼接 |
| `s2d/data/sparse.py` | 从有效像素中均匀、无放回采样，默认严格取 500 点，有效点不足时取全部 |
| `s2d/data/transforms.py` | 提供基线接口所需的缩放、裁剪、旋转、翻转、归一化和 Lighting |

当前这四个文件是按仓库接口独立实现的版本，没有从之前的 ZIP 文件直接引入。无效点处理在模型内部完成，Dataset 对外提供 RGB 和零填充的米制稀疏深度。

GitHub 缺少原始数据源码，训练增强的具体数值无法逐值对照。本次实现选择如下，团队合并时需要统一使用同一套数据处理。

- 训练时先缩放至高 250，再旋转 ±5°、随机放大 1–1.5 倍、中心裁剪，并以 50% 概率水平翻转。随机放大时深度除以对应倍数。
- 验证时缩小至原图一半，再中心裁剪到 `228×304`。
- 训练时亮度、对比度和饱和度的抖动系数为 0.6–1.4。Lighting 由配置控制，本次实验关闭 Lighting。

## 二 修改了哪些原文件

| 文件 | 改动 | 原因 |
|---|---|---|
| `s2d/models/sparse2dense.py` | 编码器前增加无效点适配器，最终输出端增加可学习输出头 | 将新增功能接入现有网络 |
| `s2d/config.py` | 增加三个配置字段、合法性检查和 `model_kwargs()` | 统一保存、传递和恢复模型配置 |
| `scripts/train.py` | 增加功能开关、实验目录标识、参数别名，使用统一模型构建接口 | 从命令行选择方案，并将配置写入 checkpoint |
| `scripts/evaluate.py` | 根据 checkpoint 配置构建模型 | 评估时恢复对应实验结构 |
| `scripts/analyze.py` | 根据 checkpoint 配置构建模型 | 对新增方案进行边界、纹理分析 |
| `s2d/models/encoder.py` | 补充 `import torch` | 修复 `copy` 初始化分支的报错 |
| `tests/test_model.py` | 修正一个输出尺寸预期值 | 原测试的尺寸计算有误 |
| `.gitignore` | 将 `data/` 改为 `/data/`，忽略本地测试环境 | 允许提交数据处理源码，同时忽略根目录数据集和本地环境 |

### 可学习最终上采样

实现位于 `sparse2dense.py` 的 `learned_head`，通过 `--learned-upsample` 开启。

原始输出路径为

```text
解码器特征 → 卷积压成单通道 → 双线性插值到目标尺寸
```

开启新功能后的输出路径为

```text
解码器特征 → 双线性插值到目标尺寸 → 3×3 卷积 → ReLU → 3×3 卷积
```

对于 ResNet50，新增卷积的通道变化为 `64→32→1`。这样网络可以在完整分辨率上学习细化输出。

DeConv3、UpConv、UpProj 直接复用仓库已有的 `s2d/models/decoder.py`，三个解码器都可以搭配上述最终输出头。

### 配置如何传递

`Config` 新增了 `learned_upsample`、`invalid_mode` 和 `invalid_value` 三个字段。训练时它们会写入配置文件和 checkpoint，评估、分析时再从 checkpoint 读取。

三个入口统一调用 `Config.model_kwargs()` 构建模型。旧 checkpoint 缺少新增字段时，使用零填充和原始最终输出头，可加载原有模型权重。

## 三 修复了哪些问题

1. `encoder.py` 调用了 `torch.no_grad()` 却缺少 `torch` 导入，`copy` 初始化分支会报错，已补齐导入。
2. `train.sh` 使用 `-a/-d/-m/-s`，Python 入口原先只接受对应长参数，已补齐这四个别名。
3. 解码器测试输入为 `2×2`，四级放大后应为 `32×32`，原预期写成 `32×40`，已修正。
4. `.gitignore` 的 `data/` 会匹配各层级同名目录，已改为 `/data/`，让 `s2d/data/` 源码能够加入版本管理。

## 四 新功能怎么运行

以下命令均在 `v2` 目录下运行，使用已安装 PyTorch、torchvision、NumPy、h5py、OpenCV、Pillow 和 matplotlib 的 Python 环境。

### 1 先确认基线设置

本次统一基线为 ResNet50、UpProj、RGBD、500 点、ImageNet 归一化、关闭 Lighting、MSE。

`Config` 和 `train.sh` 沿用 GitHub 默认值，其中损失为 L1、Lighting 强度为 0.1。新增的 `task2.py` 会显式传入 `--criterion mse --lighting-std 0`。直接运行 `scripts.train` 时，也要带上这两个参数。

### 2 预览实验命令

列出全部 24 组组合的命令。

```bash
python -m scripts.task2
```

预览某一组，例如只更换为 UpConv，或使用稀疏卷积前端。

```bash
python -m scripts.task2 --experiment t2_upconv_bilinear_zero
python -m scripts.task2 --experiment t2_upproj_bilinear_sparse
```

实验名按 `t2_解码器_最终输出头_无效点处理` 组成。解码器可选 `deconv3/upconv/upproj`，最终输出头可选 `bilinear/learned`，无效点处理可选 `zero/sentinel/mask/sparse`。基线名称为 `t2_upproj_bilinear_zero`。

### 3 统一代码后启动指定实验

准备 NYU 数据，目录结构为 `nyudepthv2/train/<scene>/*.h5` 和 `nyudepthv2/val/<scene>/*.h5`。每个 HDF5 包含 `rgb` 和 `depth`，分别为 CHW 图像与 HW 米制深度，深度 0 表示无效。

将命令中的 `/path/to/nyudepthv2` 替换成实际路径。下面三条命令分别启动基线、mask 方案和可学习上采样搭配稀疏前端的方案。

```bash
python -m scripts.task2 --experiment t2_upproj_bilinear_zero --run --data-root /path/to/nyudepthv2
python -m scripts.task2 --experiment t2_upproj_bilinear_mask --run --data-root /path/to/nyudepthv2
python -m scripts.task2 --experiment t2_upproj_learned_sparse --run --data-root /path/to/nyudepthv2
```

可以在命令后追加 `--workers 4`、`--seed 1`、`--out-dir /path/to/runs` 等原训练参数，脚本会传给 `scripts.train`。

### 4 直接使用训练入口选择功能

| 参数 | 作用 |
|---|---|
| `--decoder deconv3`、`--decoder upconv`、`--decoder upproj` | 选择解码器 |
| `--learned-upsample` | 开启可学习最终输出头，省略时使用原始输出头 |
| `--invalid-mode zero/sentinel/mask/sparse` | 选择一种无效点处理方式 |
| `--invalid-value -1` | 设置 sentinel 的负数标记，取值必须为有限负数 |

例如，使用 UpProj、负值标记和本次统一的损失及 Lighting 设置。

```bash
python -m scripts.train --data-root /path/to/nyudepthv2 --decoder upproj --criterion mse --lighting-std 0 --invalid-mode sentinel --invalid-value -1 --run-name t2_sentinel
```

使用 UpProj、稀疏卷积前端和可学习最终上采样。

```bash
python -m scripts.train --data-root /path/to/nyudepthv2 --decoder upproj --criterion mse --lighting-std 0 --invalid-mode sparse --learned-upsample --run-name t2_sparse_learned
```

`d` 和 `rgbd` 模态支持全部四种无效点处理模式，`rgb` 模态只支持 `zero`。

### 5 训练后的评估和分析

将 `runs/<run>/best.pth.tar` 替换成实际 checkpoint 路径。两个入口会读取 checkpoint 的配置，自动恢复对应的无效点处理方式和最终输出头。

```bash
python -m scripts.evaluate --checkpoint runs/<run>/best.pth.tar --save-samples 8
python -m scripts.analyze --checkpoint runs/<run>/best.pth.tar
```

默认分别输出到 checkpoint 旁的 `eval/` 和 `analysis/`，前者包含指标、深度可视化和误差图，后者包含边界与纹理分桶结果。

## 五 已完成的验证

执行下面的命令可以运行测试。

```bash
python -m unittest discover -s tests -v
```

2026 年 11 月 8 日的完整测试报告为 59 项测试整体 OK，真实 NYU 数据检查因本地缺少数据跳过。已检查的内容包括

- 全部 24 组组合的 ResNet18 前向、反向传播和权重重载。
- ResNet50、UpProj、稀疏前端与可学习输出头在 `228×304` 分辨率下的前向计算。
- 合成 HDF5 数据的 train/val 预处理、三种输入模态、500 点采样、0 点输入和固定种子复现。
- 使用临时 checkpoint 运行评估与边界、纹理分析。

验证使用 CPU 和合成数据，尚未启动正式训练。当前结论是功能和接口已通过这些检查，精度表现需要后续统一实验确认。
