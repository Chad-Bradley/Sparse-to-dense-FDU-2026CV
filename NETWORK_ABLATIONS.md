# 网络结构优化与消融实验

分支：`feat/network-ablation`，来自团队仓库 `main`。

## 环境与数据

已在 Windows 的 `cv` conda 环境检查 Python 3.10、PyTorch 2.1.0+cu121、
torchvision 0.16.0+cu121；本机补装了 h5py。不要使用缺少 torchvision 的系统 Python。
服务器新环境可以使用以下命令；CUDA 12.1 是此处验证的安装组合，可按服务器调整：

```bash
conda create -n s2d python=3.10 -y
conda activate s2d
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`main` 已包含数据加载源码，但不包含 NYU h5 数据。数据目录应为
`<data-root>/{train,val}/<scene>/*.h5`，每个文件包含 `rgb`（3×480×640 uint8）
与 `depth`（480×640 float，单位米，0 表示无效）。以下命令从项目根目录运行。

## 三个独立开关

| 改造 | Python 参数 | train.sh 环境变量 | 默认 |
|---|---|---|---|
| 瓶颈多尺度上下文 | `--multiscale aspp` | `MULTISCALE=aspp` | `none` |
| U-Net 跳连 | `--unet-skips` | `UNET_SKIPS=1` | 关闭 |
| 轻量空间传播 | `--refiner cspn-lite` | `REFINER=cspn-lite` | `none` |
| 传播核 / 轮数 | `--prop-kernel 3 --prop-steps 3` | `PROP_KERNEL=3 PROP_STEPS=3` | 3×3 / 3 |
| 关闭稀疏点回填 | `--no-preserve-sparse` | `PRESERVE_SPARSE=0` | 回填开启 |

多尺度模块为 1×1 与空洞率 1/2/3 的 3×3 四分支，拼接投影后残差融合。
跳连仅支持 `upproj`：四级依次接 layer3、layer2、layer1 和 stem；跳连特征
投影和插值到对应解码层尺寸，再拼接卷积，兼容奇数输入尺寸。

精炼使用末级解码特征、粗输出、稀疏深度和有效掩码预测邻域权重。
权重经 tanh 和绝对值和归一化，中心项作用于固定粗输出；边界复制填充。
初始邻域权重为零，因此关闭稀疏点回填时初始精炼等于粗输出。
启用回填时仅使用**输入观测**，不从监督目标取点。RGB 模态无观测点时正常传播。
精炼要求输入与输出的空间尺寸一致，以免插值伪造稀疏观测。

本模块是 **CSPN-lite**，不是完整 CSPN++ 复现；不包含论文的自适应传播核和迭代选择。
参考：https://ojs.aaai.org/index.php/AAAI/article/view/6635 。

全部关闭保留原始模型前向与参数名称，旧 checkpoint 可直接严格加载。
`model(x)` 仍返回深度张量，`model.forward_outputs(x)` 提供 `coarse`、`refined`。

## 单组训练与阶段控制

先训练对应结构的主网络。下例开启多尺度和跳连，基线只需去掉这两个参数：

```bash
python -m scripts.train --data-root /data/nyudepthv2 --out-dir runs \
  --multiscale aspp --unet-skips --epochs 15 --run-name ms_skip_backbone

python -m scripts.train --data-root /data/nyudepthv2 --out-dir runs \
  --multiscale aspp --unet-skips --refiner cspn-lite \
  --init-checkpoint runs/ms_skip_backbone/best.pth.tar \
  --refiner-only-epochs 3 --epochs 8 --refiner-lr 0.001 --joint-lr 0.0001 \
  --prop-kernel 3 --prop-steps 3 --run-name ms_skip_refined
```

精炼训练中的 `--epochs` 是新实验的总轮数：上例为 3 轮仅训精炼头 + 5 轮联合微调。
冻结阶段主网络参数和 BN 统计均固定；联合阶段重建 SGD 优化器并恢复主网络训练。
每阶段学习率独立计步，每 `--lr-step` 轮乘 `--lr-gamma`。
从随机主网络直接联合训练可显式使用 `--refiner-only-epochs 0`，不能用它冒充冻结实验。

`--init-checkpoint` 开始新实验，仅接受主网络 checkpoint，要求主网络结构和输入配置一致，
不恢复优化器。加载结果写入 `initialization.json`。不允许跳过缺失的主网络权重。
相同方式可开始无精炼的继续训练对照，学习率使用 `--joint-lr`。

`--resume` 完整续训，使用 checkpoint 配置，恢复优化器、训练阶段和随机状态；
不允许与 `--init-checkpoint` 同用。随机状态恢复适用于相同设备和 worker 配置的 epoch 边界。
旧 checkpoint 没有随机状态时仍能续训，但无法恢复之前的随机流。

```bash
python -m scripts.train --resume runs/ms_skip_refined/last.pth.tar
```

`train.sh` 还支持 `INIT_CHECKPOINT`、`REFINER_ONLY_EPOCHS`、`REFINER_LR`、`JOINT_LR`、
`VAL_SAMPLING_SEED`，Python 参数可追加到脚本末尾。Windows 建议直接调用 Python 入口。

## 八组消融与结果

```bash
# 默认只打印命令，不训练；4 组主网络 + 4 组精炼 + 4 组继续训练对照
python -m scripts.ablate --data-root /data/nyudepthv2 --out-dir runs/ablations

# 确认服务器数据与资源后执行，seed/prefix 可按重复实验修改
python -m scripts.ablate --data-root /data/nyudepthv2 --out-dir runs/ablations --execute
```

固定 ResNet50、UpProj、RGBD、500 点、L1、image 指标、seed 与验证采样种子。
四组主网络覆盖多尺度/跳连的所有组合；各自主网络 checkpoint 再用于有精炼和无精炼继续训练。
无精炼对照继续微调 5 轮，与精炼实验的联合微调轮数一致。
这不等于总计算量完全匹配：精炼实验还多了 3 轮冻结头训练，应在报告中标明。
重新执行请换 `--prefix`，脚本会拒绝复用已存在的实验目录。

`config.json` 保存全部开关和初始化路径；`experiment.json` 记录参数量和试跑截断；
`metrics.csv` 记录阶段、学习率、耗时和指标。每阶段保存 `<stage>_best.pth.tar`，
例如 `refiner_only_best.pth.tar` 与 `joint_best.pth.tar`；`best.pth.tar` 是整个新实验最优，
不保证来自联合阶段。报告应分别列出原主网络、仅训精炼头后、联合微调后的结果。

```bash
python -m scripts.evaluate --checkpoint runs/ms_skip_backbone/best.pth.tar
python -m scripts.evaluate --checkpoint runs/ms_skip_refined/refiner_only_best.pth.tar
python -m scripts.evaluate --checkpoint runs/ms_skip_refined/joint_best.pth.tar
# 同一个精炼 checkpoint 的粗预测，可判断联合微调后的主网络变化
python -m scripts.evaluate --checkpoint runs/ms_skip_refined/joint_best.pth.tar \
  --prediction coarse --output-dir runs/ms_skip_refined/eval_coarse
python -m scripts.analyze --checkpoint runs/ms_skip_refined/joint_best.pth.tar
```

验证集按样本索引和 `--val-sampling-seed` 固定采样，不依赖 epoch 或 worker。
这修正了旧验证流程每轮重新随机取点的行为；历史指标应使用相同新采样重新评估再比较。
GPU 运算仍可能存在非确定性；正式报告建议 seed 0/1/2 分别运行后汇总均值与标准差。

## 服务器小样本试跑

真实数据试跑在服务器进行。本机单测覆盖前后向和训练阶段，不代表精度提升。
下例完整穿过主网络训练、冻结、联合阶段；ResNet18、较小 batch 用于检查流程。

```bash
python -m scripts.ablate --data-root /data/nyudepthv2 --out-dir runs/smoke \
  --prefix smoke --arch resnet18 --batch-size 2 --workers 0 --no-pretrain \
  --limit-train 4 --limit-val 2 --backbone-epochs 1 \
  --refiner-epochs 1 --joint-epochs 1 --execute

python -m scripts.train --resume runs/smoke/smoke_ms0_skip0_refined/refiner_only_best.pth.tar \
  --limit-train 4 --limit-val 2
python -m scripts.evaluate --checkpoint runs/smoke/smoke_ms0_skip0_refined/joint_best.pth.tar \
  --data-root /data/nyudepthv2 --save-samples 2
```

试跑应产生全部模型 checkpoint 和阶段日志；续训从冻结阶段末尾恢复时应进入联合阶段。
训练时的 limit 参数不保存在 Config；续训小样本试跑请同样指定 `--limit-train 4 --limit-val 2`。
