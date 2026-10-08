# CEE + CTSA 短测（RTX 4090 单卡）

目标：原日志约 11.2 秒/步、7.4 小时/轮；第一阶段争取耗时减半。
本工具仅准备测量，不宣称已达到加速目标。

## 同步文件

将下列文件按原目录结构同步到 Linux 项目根目录：

- semiearth.py
- model/semseg/vlm_pp.py
- util/benchmark_runtime.py
- scripts/benchmark_ctsa.py

不用覆盖 configs/loveda.yaml，保留服务器已经修好的 Qwen 路径。
脚本使用服务器当前配置及现有数据、Qwen/SAM 权重。

## 运行

先正常停止同一 GPU 上的其他训练；不要并行覆盖正在读取的 latest.pth。
进入 Linux 项目后运行：

```bash
cd /home/a8/qhf/SemiEarth-master05
/home/a8/anaconda3/bin/python scripts/benchmark_ctsa.py loveda 5_100 \
  --checkpoint exp/loveda/semiearth_cee_ctsa/dinov2_small/5_100/latest.pth
```

默认依次执行：batch2（5步预热+20步测速）、batch4（同上）、
详细剖析（5步预热+3步剖析，并对3批相同教师输入比较batch2和batch4净化结果）。
按原耗时粗估约10–15分钟，模型加载和实际输出长度会影响时长。
同一随机种子控制数据顺序及增强；每组从同一检查点恢复学生、EMA、优化器和进度。
短测会更新内存中的模型来复用实际训练路径，不写任何模型检查点。
若两种净化输出存在差异，独立测速过程的后续模型轨迹也可能不同；因此必须结合配对输出检查解释速度。

检查点必须包含model/model_ema/optimizer/epoch，且已越过CTSA预热并尚未完成全部训练。
如检查点仍在CTSA预热期，工具明确退出；不要通过缩短epochs来绕过。
检查点的训练设置须与服务器配置一致，特别是数据划分、batch_size和CTSA设置。
不要使用 --init-checkpoint，因为它会重置训练进度。

启动器固定使用静态 rendezvous 与 127.0.0.1，避免数字主机名导致连接超时。
默认端口29511；如果端口被占用，可加 --port 29512。

可选参数：--gpu 0、--config configs/loveda.yaml、--steps 50、--detail-steps 5。
--output 可以指定一个尚不存在的独立目录。

## 输出与判断

输出在 exp/benchmarks/时间戳/：

- benchmark_summary.json：完整配置、硬件、输入/权重哈希、各组统计及配对结果。
- report.md：阅读版对比。
- batch2.log、batch4.log、detail_compare.log：原始日志。
- 各组 steps.jsonl：逐步耗时与显存。
- detail_compare/steps.jsonl：CEE边缘/统计、父类筛选/融合、Qwen生成、SAM图像编码/逐框预测。

优先发回 benchmark_summary.json、report.md 和 detail_compare/steps.jsonl。
详细剖析会同步GPU，嵌套方法时间互相包含，不能求和；它不用于速度验收。
Qwen batch时间包含预处理/生成/解码；生成与解码另列。剩余部分可用于判断预处理是否明显。
生成长度使用token slots（包含批内padding），代表解码工作量，不是自然语言有效token数。
短测记录Qwen实际生成次数/图像数以及SAM逐框调用次数；跳过净化的批次不会误算调用。

batch4发生显存不足时，既有运行器可能回退到逐图处理，报告会标明；不要据此认为batch4生效。
配对比较在同一教师输出和同一图像上计算：标签差异、置信度误差、CEE监督掩码差异。
标签/掩码必须一致，置信度最大误差阈值1e-6；少量样本一致不等于最终mIoU一致。
若输出有变化，先检查差异，不自动把batch4写入正式配置。
成功短测后再进行更长的计时及精度验证，决定是否采用。

## 本地验证与能力边界

本地Windows只有CPU测试环境，可以运行：

```text
python -m unittest test_benchmark_runtime
python test_cee.py
```

本地不能代替4090上的Qwen/SAM/CUDA验证。未运行服务器短测前，不能保证提速倍数。
本轮仅增加显式短测入口与计时钩子，CEE阈值、筛选、融合公式和正常训练参数没有改变。
