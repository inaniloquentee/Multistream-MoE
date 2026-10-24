# 验证记录（2026-10-25）

## 环境和复现

远端目录 `/home/ljj/Multistream-MoE/ljj`；Python `/home/ljj/vllm-pr-59863/.venv/bin/python3`，PyTorch `2.14.0+cu130`，Transformers `5.19.0`。Qwen 路径直接使用官方 Transformers 的 `Qwen3ForCausalLM` 和 SDPA，不依赖 nanoGPT；小从零训练参考才需要 `NANOGPT_PATH`（本次 checkout `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`）。测试运行于远端头节点 CPU，BF16 小配置检查调用头节点 A10，并非 H20 的 Qwen 训练结果。

权重/tokenizer：`Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。工程数据：FineWeb-Edu `sample-10BT` viewer 前 1000 行，观察到的 dataset revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`；892/108 个互斥文档，939516/123095 个训练/验证 token；`train.bin` SHA256 `8b79ff0bb57f740ece99de3ebd137780a5f6d5f35b5080d41db143191e1f148e`，`val.bin` SHA256 `564f624077ff668dea3d87bd65f2faa5c6dc9b2a25a2491091dc911db2b249d2`。这些哈希验证本次输入，但 viewer 不是锁定快照，不可用作正式质量比较。

```bash
cd /home/ljj/Multistream-MoE/ljj
PYTHON=/home/ljj/vllm-pr-59863/.venv/bin/python3
$PYTHON -m unittest -v test_qwen
$PYTHON check_qwen_checkpoint.py --model cache/qwen3-0.6b-base --data data/qwen-fineweb-proxy --modes ABCD
NANOGPT_PATH="$PWD/_upstream/nanoGPT" $PYTHON -m unittest -v test_reference
```

真实 checkpoint 的 CPU FP32 固定 8-token 输入，`max_logit_abs_error` 对原始 dense Qwen：

| 组别 | 最大 logits 绝对误差 | 最终流中心化 RMS | 最终流均值 RMS | 参数量 |
|---|---:|---:|---:|---:|
| A | 2.479553e-5 | 0 | 22.135769 | 1,124,675,598 |
| B | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,005,648 |
| C | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,235,248 |
| D | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,464,848 |

初始等价是精度范围内的数值结果，并非数学意义的浮点逐位相等。小配置 12 项测试通过，包括 B/C/D 初始状态和路由器一致性、D 第二个 MoE 的独立读取梯度、有限差分与 A10 上 BF16/FP32 对照。真实 checkpoint 尚未执行训练反向。

## 尚未通过的门槛

2026-10-25 查询 `sinfo -N -p h20`：`h20-01` 至 `h20-05` 均 `down*`，`h20-06` 为 `drained*`；首节点 Slurm 原因 `Not responding`，无可分配 H20。因此 **Qwen 五组 32 步 BF16 训练没有运行**；更长的固定语料、多种子质量对照、EP、通信 profiling、融合内核都未执行。不能由初始化等价、较小配置测试或代理集推断 D 优于 C/B、成本收益或 idea 正确性。待节点可用时先从该目录执行：

```bash
mkdir -p results
sbatch -p h20 --gres=gpu:1 --time=04:00:00 \
  --output="$PWD/results/qwen-slurm-%j.log" run_qwen.sh
```

该命令依次运行 `dense,A,B,C,D` 的 32 步诊断，日志写到被 `.gitignore` 忽略的 `results/`。提交前须确认实际节点、权限和时限；**不应把计划命令写成已完成结果**。正式 Q2 的数据快照、训练预算和判据见 `EXPERIMENT.md`。
