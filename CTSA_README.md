# SemiEarth + CEE + CTSA：修改说明与运行方式

## 1. 保留了什么

`util/cee.py`、`model/semseg/vlm_pp.py`、`test_cee.py` 保持原文件不变。已有 CEE 配置值全部保留，包括 LoveDA 的 `edge_conf_thresh: 0.75`、Potsdam 的 `0.85`、边缘宽度、VLM 边缘权重、目标像素选择和置信度融合。

CTSA 的辅助损失直接使用训练入口已有的 CEE `conf_mask`，没有用统一的 0.95 阈值重新过滤边缘。CTSA 门控仅控制教师特征注入，不修改 CEE 或净化标签。新增的 Qwen 批量包装最终仍调用原 `get_qwen_purify` 执行 CEE/SAM/标签融合。

## 2. CTSA 的实际接法

- 取 DINOv2 最后一组 patch tokens，学生为 Q，停止梯度的教师为 K/V。
- 每个样本独立计算通道交叉注意力，投影维度 128、4 个头，残差系数 0.1；这是方案中的工程适配形式。
- 共享学生编码器及 DPT 前三组特征的投影，只增加末层重构和辅助解码。
- 教师在一次已有前向中同时返回 logits 和末层特征，不重新跑编码器。
- 原学生预测分支始终存在，测试只执行这一分支。教师不包含 CTSA，EMA 按参数名称更新编码器和解码器，避免新增参数导致错位。
- CutMix 使用原来的 `flip(0)` 配对，同时对教师特征和原始教师标签作对齐。跨越 CutMix 接缝的 patch 关闭注入；ViT 编码后再混合特征仍是上下文近似。
- 默认 `ctsa_gate: agreement`，依据 CEE 接受掩码、净化置信度及净化前后标签一致性构建门控。包含纠正像素的 patch 保守关闭特征注入，但可信纠正像素仍可参与分割监督。

损失为 `原损失 + ct_weight * 辅助交叉熵`，主损失保持 `(loss_x + loss_u_s) / 2`。辅助损失沿用全部非 ignore 像素归一化。前 10% 训练关闭辅助计算；随后 10% 线性增加至 0.1。预热跳过在所有 rank 上一致，DDP 保留 `find_unused_parameters=True`；启用 CTSA 后，某个 rank 的门控全零也不单独跳过辅助前向。

## 3. 速度优化

| 修改 | 减少的开销 | 说明 |
|---|---|---|
| 学生 AMP | 学生前向/反向计算和激活显存 | 自动优先 BF16，不支持时使用 FP16 + GradScaler；教师默认仍 FP32 |
| DINO 的 PyTorch SDPA | 无 xFormers 时显式构建完整空间注意力矩阵的开销 | 注意力定义不变；CUDA 会选择可用后端 |
| 移除每步 barrier 和强制 CUDA synchronize | 不必要的等待 | DDP 梯度同步仍然保留；诊断计时除外 |
| GPU 累积日志统计、降低日志频率 | 多次 `.item()` 和 TensorBoard 写入 | CEE 的统计定义不变 |
| 只生成实际使用的一份强增强 | PIL 变换、worker 传输、内存 | 数据集默认旧接口仍保留，训练入口明确选择单强增强模式 |
| 持久化 workers、预取、异步传输 | 每轮启动 worker 和数据搬运 | 可通过配置关闭或调整 |
| 验证统计在 GPU 累积，结尾一次同步 | 逐图 NumPy 往返及三次 all_reduce | mIoU 类别口径和验证频率保留 |
| Qwen 批量推理，默认每次 2 张 | 串行生成开销 | 保持 prompt、256 token 上限、图像分辨率和 CEE 目标选择；显存不足时回退串行 |
| DDP 下 Qwen 绑定当前 rank GPU | 多个进程同时自动跨多张卡分配模型 | 单进程仍保留 auto，可显式配置 `qwen_device_map` |

**CTSA 本身会增加训练计算量，整体提速幅度需要在训练机测量。** 本机没有 CUDA、数据集和 Qwen/SAM 权重，不能给出真实每轮耗时或精度结论。Qwen 批处理、SDPA 和混合精度可能有浮点差异，因此不保证真实生成文本逐 token 完全相同；CEE 算法保持不变。

没有为提速降低 CEE 的调用频率、提高 `min_edge_pixels`、关闭 SAM、缩短训练、降低裁剪尺寸或启用跨图像缓存。这些都会改变实验条件。

## 4. 启动训练

在 Linux 训练机的项目根目录执行。配置中的 `data_root`、LoveDA 的 `qwen_model_name` 仍是你原来的值；迁移机器时请改成对应路径。

```bash
# 从 DINOv2 预训练权重开始，LoveDA 5%，单卡示例
bash scripts/train_ctsa.sh loveda 5_100 1 29501

# 4 卡示例
bash scripts/train_ctsa.sh loveda 5_100 4 29501

# 从已有 CEE 模型初始化；第五个参数是你自己的权重路径
bash scripts/train_ctsa.sh loveda 5_100 1 29501 /path/to/cee/best.pth
```

新脚本保存到 `exp/<dataset>/semiearth_cee_ctsa/dinov2_small/<split>`，与原实验目录分开。显卡数应填写实际使用数量；脚本不会自行修改 `CUDA_VISIBLE_DEVICES`。

从 CEE 初始化使用 `--init-checkpoint`，只加载学生/教师权重，CTSA、优化器和训练日程重新开始，必须使用新保存目录。已有 CTSA 实验仍会自动读取同目录 `latest.pth`，恢复优化器、AMP scaler 和 epoch。CTSA 开关与旧检查点不一致时会明确报错，避免误把旧实验当作完整续训。

默认测试接口不变：

```bash
python test.py --config configs/loveda.yaml --checkpoint exp/loveda/semiearth_cee_ctsa/dinov2_small/5_100/best.pth
```

测试加载器只排除 `ctsa.*` 参数，编码器/解码器缺失或不匹配仍报错。

## 5. 配置与对照

```yaml
use_ctsa: true
ctsa_gate: agreement       # none / confidence / agreement
ctsa_gate_statistics: false  # true 时也筛选通道相关性统计
ctsa_weight: 0.1
ctsa_warmup_ratio: 0.1
ctsa_ramp_ratio: 0.1
amp: true
amp_dtype: auto
teacher_amp: false
eval_amp: false
qwen_batch_size: 2         # 显存不足或要对照串行结果时设 1
num_workers: 4
val_num_workers: 2
profile_steps: 5           # 仅首轮前 5 步详细同步计时；设 0 完全关闭
```

先比较 `use_ctsa: false` 与 `true`，两组均使用同一套提速设置；再比较 `ctsa_gate: none / confidence / agreement`。仅置信度模式不会剔除被纠正 patch，以免混入一致性规则。所有实验使用不同保存目录。

若要求 FP32 对照，设 `amp: false`；教师/验证默认没有开 AMP。移除无用增强会改变后续随机数流，AMP 也会改变数值轨迹，因此新旧版本不保证相同 seed 的训练轨迹逐步一致。比较精度应保持版本与运行条件一致并重复种子。

## 6. 怎样看真实瓶颈

前几步日志会显示：

- `teacher_ms`：数据传输和教师前向。
- `vlm_ms`：CEE 统计、Qwen、SAM 和净化调用。
- `student_forward_ms`：CutMix、CTSA（启用后）、学生前向及损失。
- `backward_optimizer_ms`：反向传播和优化器。
- `ema_ms`：统计与 EMA 更新。

每轮另外打印训练总秒数和“学生＋EMA 验证”总秒数。首轮前几步的 CTSA 尚在预热期，不包含辅助解码；训练总时间仍覆盖后续 CTSA 开启阶段。计时并非全程无开销，`profile_steps: 0` 可关闭详细同步计时。

如果 `vlm_ms` 占绝大多数，AMP 对总时间的影响会有限。先确认 `qwen_batch_size: 2` 不发生 OOM，再按显存尝试 4；不要用不同图像共用空间缓存来换速度。

## 7. 验证记录

已在独立 CPU 环境（PyTorch 2.9.0）完成：

- 11 项自动测试：Q/K/V 梯度、教师停止梯度、零门控、跨样本隔离、CutMix 与门控、CEE 接受掩码、指标等价、真实 DINOv2-small+DPT 前后向、EMA/权重加载、权重日程、单强增强输出一致、模拟 Qwen 返回值下的批处理/原 CEE 行为对照。
- 原 `test_cee.py` 的 CPU 项通过；CUDA 项因本机无 CUDA 跳过。
- 固定权重下，重构后的普通 DPT 路径与原项目 DPT 输出逐元素一致；SDPA 与手工注意力在浮点容差内一致。
- 训练入口参数检查及 Python 语法编译通过。

双进程测试脚本 `test_ctsa_ddp.py` 已提供，本地 PyTorch Windows CPU 构建报 `unsupported gloo device`，未能完成分布式运行验证。请在 Linux 训练环境运行：

```bash
python -m unittest test_ctsa -v
python test_cee.py
python test_ctsa_ddp.py
```

尚未验证：目标 CUDA/NCCL、真实 Qwen/SAM 批推理、完整数据集训练、实际提速比例及精度变化。模拟 Qwen 测试只证明调度及 CEE 后处理一致，不代表真实模型批量生成必定逐 token 相同。

原 VLM 的置信度归一化和显式缓存语义保持原样；此前方案提及的有效净化率问题不在此次 CTSA/性能改动中修正，避免改变 CEE 实验基础。

参考：[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[Qwen2.5-VL Transformers 4.57.1 文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)。
