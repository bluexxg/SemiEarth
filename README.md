# SemiEarth + CEE + CTSA 完整实验项目（2026-09-30）

本包是“当前已完成修复的代码整合版”，不是定位质量已全部通过的最终论文模型。包含完整源码、数据划分、配置、训练/评估脚本、短测、诊断、回归测试及实验记录。无需叠加此前零散补丁。

## 已合并内容
- DINOv2-Small + DPT 教师—学生模型与 EMA。
- 原 CEE 边缘选择、置信度融合和边缘/非边缘监督阈值保持不变。
- CTSA 通道交叉注意力、可靠性门控、CutMix 特征对齐及共享 DPT 辅助路径。
- AMP、Qwen 批处理、单强增强数据路径和既有运行时优化。
- Qwen JSON 解析修复：bbox_2d + label/class_name，兼容旧文本；仅恢复完整记录，不编造截断坐标。
- benchmark_ctsa.py 使用静态127.0.0.1通信；新增解析器和启动器哈希记录。
- diagnose_vlm.py 保存原始回答、截断记录、SAM调用及净化改动数量。
- 新增check_project.py检查实际数据路径、非空权重文件及CUDA。

## 本包与旧项目的区别
- configs/loveda.yaml 和 potsdam.yaml 使用相对Qwen目录 pretrained/qwen2_5_vl_3b_instruct。
- qwen_batch_size 默认4，是已有4090诊断运行成功的候选设置。训练batch_size仍为4，不是将训练batch翻倍。修复后完整路径的batch2/4精度等价与长测仍需验证。
- 新启动器可用PYTHON_BIN指定Python、SAVE_PATH指定新实验目录、CONFIG指定配置。遇到空latest.pth或初始化与断点混用时提前退出。
- 原始 requirements.txt 保留为依赖清单；它并非服务器已核实的完整锁文件。服务器实测Torch为2.9.1+cu128而清单为2.9.0，不能称为环境完全复现。
- 不含Git历史、IDE缓存、旧实验输出、训练权重、数据集和大型预训练模型。复用服务器已有环境，不必重新安装全部依赖。

## 1. 在服务器解压
将ZIP上传到 /home/a8/qhf 后执行：
```bash
cd /home/a8/qhf
unzip SemiEarth_CEE_CTSA_20260930.zip
cd SemiEarth_CEE_CTSA_20260930
```
压缩包已有一层同名项目目录，不要再套一层。

## 2. 连接已有数据与模型（不复制大文件）
新项目中的pretrained和datasets仅有说明文件。执行：
```bash
ln -s /home/a8/qhf/SemiEarth-master05/pretrained/dinov2_small.pth pretrained/dinov2_small.pth
ln -s /home/a8/qhf/SemiEarth-master05/pretrained/sam_vit_b_01ec64.pth pretrained/sam_vit_b_01ec64.pth
ln -s /home/a8/qhf/SemiEarth-master05/pretrained/qwen2_5_vl_3b_instruct pretrained/qwen2_5_vl_3b_instruct
ln -s /home/a8/qhf/SemiEarth-master05/datasets/LoveDA datasets/LoveDA
```
若链接已存在，不重复创建；原文件真实路径不同则修改来源。不要删除作为链接来源的旧模型或数据目录。Potsdam另按实际目录设置data_root；本轮服务器结果仅覆盖LoveDA。

## 3. 检查运行条件
```bash
/home/a8/anaconda3/bin/python scripts/check_project.py loveda 5_100
```
检查所有划分引用文件是否存在、模型文件非空以及CUDA可用；不加载模型，因此不能证明权重完整或依赖全都兼容。

## 4. 先做3步诊断（建议先执行）
停止同卡其他训练，保持所用检查点不再被写入：
```bash
/home/a8/anaconda3/bin/python scripts/diagnose_vlm.py loveda 5_100 \
  --checkpoint /home/a8/qhf/SemiEarth-master05/exp/loveda/semiearth_cee_ctsa/dinov2_small/5_100/latest.pth \
  --port 29511
```
它仅在内存中执行训练步骤，不保存模型检查点。输出在exp/vlm_diagnostics。现有checkpoint已过CTSA预热，可继续用于诊断，不必为此从头训练。

## 5. 正式训练入口（完成定位质量核查后使用）
从DINOv2预训练权重开始：
```bash
PYTHON_BIN=/home/a8/anaconda3/bin/python \
SAVE_PATH=exp/loveda/cee_ctsa_parser_fixed/5_100/run01 \
bash scripts/train_ctsa.sh loveda 5_100 1 29501
```
首次运行目录应没有latest.pth；之后相同命令会恢复该目录latest.pth。若想开始另一场独立实验，改成run02。不要将示例/path/to/cee/best.pth原样加入命令。

已有旧CEE权重时，第五参数只能填真实、可加载的检查点，且必须使用新的SAVE_PATH；它是初始化，不是保留优化器和训练进度的续训。多卡示例将第三参数1换成4，但本包本机未验证多卡CUDA。

## 6. 修复后重新短测
```bash
/home/a8/anaconda3/bin/python scripts/benchmark_ctsa.py loveda 5_100 \
  --checkpoint /home/a8/qhf/SemiEarth-master05/exp/loveda/semiearth_cee_ctsa/dinov2_small/5_100/latest.pth \
  --port 29511 --steps 50 --detail-steps 5
```
诊断/短测与正常训练不要同时占用同一GPU。输出一致检查不能替代mIoU验证；报告中的purification_observed只表示Qwen生成发生，不等于净化正确，应结合诊断的SAM与像素修改记录。

## 当前验证状态与剩余问题
25项CPU单元测试通过，CEE自检状态见VALIDATION.md。全项目Python语法通过。服务器已验证解析修复版本：12张图解析53框，SAM图像编码12次、框预测53次，3步修改2058个类别像素及118099个置信度像素。

尚未修复/验证：7/12回答被截断、多类别重复框、ignore被当作定位类别、框坐标与processor缩放空间的对应。解析成功不等于类别定位正确；不能宣称修改的2058像素全部被正确纠正。后续提示词、类别策略或坐标调整应作为独立实验变体，本包没有静默加入这些未验证改变。

原batch2→4的41.8%耗时减少是在解析失败状态测得，不能作为完整净化修复后的性能结论。修复后仅两步约7.30秒，不构成稳定速度证明。建议先完成定位可视化和质量核查，再长测及正式精度实验。

## 文件入口
- semiearth.py：训练主流程。
- model/semseg/dpt.py、ctsa.py：网络与辅助模块。
- model/semseg/vlm_pp.py、vlm_runtime.py：净化与批处理。
- util/grounding_parser.py：检测框解析。
- configs/、splits/：配置及划分。
- scripts/train_ctsa.sh、benchmark_ctsa.py、diagnose_vlm.py、check_project.py：训练、短测、诊断、预检。
- docs/experiment_records/：修复依据与服务器诊断报告。
- MANIFEST.sha256.json：包内文件完整性校验（不含该清单自身）。
