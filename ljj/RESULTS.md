# 验证记录（2026-10-25）

2026-10-27 的 H20 读取通道机制排查、强流对比 C/D 配对消融和原始日志见 [MECHANISM.md](MECHANISM.md)；这是探索性诊断，不改变下文三种子预注册比较的结论。

## 环境和复现

远端目录 `/home/ljj/Multistream-MoE/ljj`；Python `/home/ljj/vllm-pr-59863/.venv/bin/python3`，PyTorch `2.14.0+cu130`。Transformers fork `https://github.com/inaniloquentee/transformers.git` 的实验分支固定提交 `c4c4605f961877852b89b97478536bb3a26b6fd1`（上游 `v5.19.0`）；实测 import 路径 `/home/ljj/Multistream-MoE/ljj/_upstream/transformers/src/transformers/__init__.py`，版本 `5.19.0`。Qwen 路径复用该 fork 的 `Qwen3ForCausalLM` 和 SDPA，不依赖 nanoGPT；小从零训练参考才需要已有 nanoGPT fork（本次 checkout `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`）。测试运行于远端头节点 CPU，BF16 小配置检查调用头节点 A10，并非 H20 的 Qwen 训练结果。

权重/tokenizer：`Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。工程数据：FineWeb-Edu `sample-10BT` viewer 前 1000 行，观察到的 dataset revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`；892/108 个互斥文档，939516/123095 个训练/验证 token；`train.bin` SHA256 `8b79ff0bb57f740ece99de3ebd137780a5f6d5f35b5080d41db143191e1f148e`，`val.bin` SHA256 `564f624077ff668dea3d87bd65f2faa5c6dc9b2a25a2491091dc911db2b249d2`。这些哈希验证本次输入，但 viewer 不是锁定快照，不可用作正式质量比较。

```bash
cd /home/ljj/Multistream-MoE/ljj
PYTHON=/home/ljj/vllm-pr-59863/.venv/bin/python3
bash scripts/bootstrap_transformers.sh
source scripts/use_transformers_fork.sh
$PYTHON -m unittest -v tests.test_qwen
$PYTHON -m experiments.check_qwen_checkpoint --model cache/qwen3-0.6b-base --data data/qwen-fineweb-proxy --modes ABCD
NANOGPT_PATH="$PWD/_upstream/nanoGPT" $PYTHON -m unittest -v tests.test_reference
```

真实 checkpoint 的 CPU FP32 固定 8-token 输入，`max_logit_abs_error` 对原始 dense Qwen：

| 组别 | 最大 logits 绝对误差 | 最终流中心化 RMS | 最终流均值 RMS | 参数量 |
|---|---:|---:|---:|---:|
| A | 2.479553e-5 | 0 | 22.135769 | 1,124,675,598 |
| B | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,005,648 |
| C | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,235,248 |
| D | 1.907349e-5 | 1.096819 | 22.135765 | 1,125,464,848 |

初始等价是精度范围内的数值结果，并非数学意义的浮点逐位相等。小配置 12 项测试通过，包括 B/C/D 初始状态和路由器一致性、D 第二个 MoE 的独立读取梯度、有限差分与 A10 上 BF16/FP32 对照。真实 checkpoint 尚未执行全参数训练反向。

## 真实 checkpoint 的低精度预检（2026-10-25，头节点 A10）

同一验证流前 17 个 token，输入长度 16，目标为后移 1 位。命令：

```bash
cd /home/ljj/Multistream-MoE/ljj
source scripts/use_transformers_fork.sh
for mode in dense A B C D; do
  /home/ljj/vllm-pr-59863/.venv/bin/python3 -m experiments.check_qwen_bf16 \
    --model cache/qwen3-0.6b-base --data data/qwen-fineweb-proxy \
    --seq-len 16 --mode "$mode"
done
```

| 组别 | FP32 NLL | BF16 NLL | BF16 对本组 FP32 最大 logits 绝对差 |
|---|---:|---:|---:|
| dense | 5.280904 | 5.290874 | 0.447740 |
| A | 5.280905 | 5.271801 | 1.429064 |
| B | 5.280904 | 5.255944 | 1.241555 |
| C | 5.280904 | 5.255944 | 1.241555 |
| D | 5.280904 | 5.255944 | 1.241555 |

D 的完整 checkpoint 在 BF16/FP32 下只解冻第一层 MoE 的写回权重和第二层 MoE 的读取权重，反向后两者均非零且有限：写回梯度范数 `0.694208/0.762909`，读取梯度范数 `0.007502/0.006284`（BF16/FP32）；各自梯度差范数 `0.100698`、`0.001356`；峰值分配显存约 6.69 GB。**这不是全参数反向、优化器一步或训练验证。** B/C/D 的 BF16 起点在该单个短样本上相同，但 dense/A 与其不同；FP32 checkpoint 初始等价不能替代训练精度的逐组起点检查。正式质量比较必须在完整固定验证窗口上重验，并以 D-C、C-B 的配对变化和 GPU 时为准。以上头节点 A10 结果不可替代 H20 性能数字。

## 2026-10-25 的节点阻塞（历史记录）

2026-10-25 02:28（上海时间）查询 `sinfo -N -p h20`：`h20-01` 至 `h20-05` 均 `down*`，`h20-06` 为 `drained*`；首节点 Slurm 原因 `Not responding`，无可分配 H20。因此 **Qwen 五组 32 步 BF16 训练没有运行**；更长的固定语料、多种子质量对照、EP、通信 profiling、融合内核都未执行。不能由初始化等价、上述单样本精度预检或代理集推断 D 优于 C/B、成本收益或 idea 正确性。待节点可用时先从该目录执行：

```bash
mkdir -p results
sbatch -p h20 --gres=gpu:1 --time=04:00:00 \
  --output="$PWD/results/qwen-slurm-%j.log" scripts/run_qwen.sh
```

该命令依次运行 `dense,A,B,C,D` 的 32 步诊断，日志写到被 `.gitignore` 忽略的 `results/`。提交前须确认实际节点、权限和时限；**不应把计划命令写成已完成结果**。正式 Q2 的数据快照、训练预算和判据见 `EXPERIMENT.md`。

2026-10-25 目录重排后，从 `ljj/` 用 `tests.*` 和 `experiments.*` 模块入口重跑了 12 项测试、真实 checkpoint 的 D 组前向及 BF16 读写梯度预检，结果与上表一致。`experiments.prepare_qwen --download-only` 离线命中原 `ljj/cache/`；`scripts/run_qwen.sh` 在非 H20 Slurm 作业中拒绝启动。其余 A/B/C 的真实 checkpoint 数值来自重排前的同一实现，未把目录变化当作新的收益实验。

## H20 恢复后的 Q1 诊断（2026-10-26，上海时间）

`srun` 实际分配 `vllm-sgp-h20-05` 的 NVIDIA H20-3e（143771 MiB）；Slurm 作业 `2847` 执行 `scripts/run_qwen.sh`，状态 `COMPLETED`、退出码 `0:0`、墙钟 `00:02:47`。权重、Transformers fork 与上文相同；训练/验证 bin 的 SHA256 也与上文一致。种子 42、batch 1、序列 128、BF16 autocast、32 步、8 个固定验证窗口，每组从同一官方 checkpoint 独立启动。下表 NLL 不含路由辅助损失：

| 组别 | step 0 | step 16 | step 32 | 训练循环秒数 | 峰值分配显存 GiB |
|---|---:|---:|---:|---:|---:|
| dense | 3.488788 | 3.506694 | 3.502082 | 3.744 | 10.28 |
| A | 3.489507 | 3.514226 | 3.506584 | 4.958 | 19.14 |
| B | 3.491964 | 3.513310 | 3.504835 | 10.129 | 19.15 |
| C | 3.491964 | 3.516354 | 3.506911 | 9.264 | 19.15 |
| D | 3.491964 | 3.514726 | 3.506586 | 10.450 | 19.15 |

B/C/D 的 BF16 起点一致；各组 32 步均完成，训练总损失和末步梯度范数有限。D 比 C 的末步 NLL 低约 0.000326，但比 B 高约 0.001751，且三组均比各自起点差。循环秒数不含初次加载与验证，不应当作端到端 GPU 时；短步数与代理数据也不支持架构优劣结论。A/dense 的 BF16 起点本来就不同，不用其差值归因于独立读写。

另在 H20 上执行 `python -m experiments.check_qwen_bf16 --mode D --seq-len 16`：FP32/BF16 NLL `5.280909/5.283858`、最大 logits 绝对差 `0.804055`；第一层写回梯度范数 `0.762903/0.713987`，第二层读取 `0.006284/0.006789`（FP32/BF16），均有限且非零。这是冻结其余权重的单样本梯度预检，不替代完整训练反向对照。

## 多种子代理集延长诊断（2026-10-26）

Slurm 作业 `2848` 在同一 `vllm-sgp-h20-05` 上执行 `scripts/run_qwen_paired.sh`，状态 `COMPLETED`、退出码 `0:0`、墙钟 `01:04:18`。种子 42/43/44 分别从同一官方 checkpoint 启动 dense/A/B/C/D，各 1,000 步、batch 1、序列 128、BF16、学习率从 `1e-5` 余弦降至 10%（**没有 Q2 规定的预热**）；每 100 步在相同的 64 个固定验证窗口上评估，即每次约 8,192 个 token。训练约 128,000 token/组。输入仍是上述 viewer 代理集，不是锁定的 Parquet 快照；这轮是额外的工程探索，**不是**第 7 节 Q2，也不是可发表的质量对照。原始逐组 JSONL 在 `evidence/h20-2026-10-26/paired/`，32 步日志在同级 `q1/`。所有 15 份日志均有 step 0、100、...、1000；所有 NLL 有限，B/C/D 在每个种子的 step 0 完全相同。

| 种子 | dense 末步 | A 末步 | B 末步 | C 末步 | D 末步 | D−C | D−B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.222867 | 3.232390 | 3.230687 | 3.231974 | 3.232255 | +0.000281 | +0.001568 |
| 43 | 3.223350 | 3.231952 | 3.232876 | 3.232639 | 3.231822 | −0.000818 | −0.001055 |
| 44 | 3.215893 | 3.222455 | 3.223064 | 3.222295 | 3.222957 | +0.000661 | −0.000108 |
| 均值 | 3.220703 | 3.228933 | 3.228876 | 3.228969 | 3.229011 | +0.000042 | +0.000135 |

C−B 的三个配对差值是 `+0.001287、−0.000237、−0.000769`，平均 `+0.000093`。D−C 两正一负，D−B 一正两负；没有一致方向，也没有达到预设的 `0.01 nats/token` 探索门槛。各组末步 NLL 均高于自己的 step 0（B/C/D 起点随种子为 `3.158586/3.158819/3.158977`）；预训练 dense 在三个种子末步均低于 B/C/D，且参数更少。**这些代理集上的负观察不是架构不可行的证明**，但不支持目前投入专用 dispatcher/kernel。

| 组别 | 参数量 | 平均 1,000 步训练循环秒数 | 峰值分配显存约 GiB |
|---|---:|---:|---:|
| dense | 596,049,920 | 72.14 | 10.28 |
| A | 1,124,675,598 | 140.02 | 19.14 |
| B | 1,125,005,648 | 249.98 | 19.15 |
| C | 1,125,235,248 | 249.84 | 19.15 |
| D | 1,125,464,848 | 252.50 | 19.15 |

循环时间剔除了验证与初次模型加载，不是端到端 wall-clock；单卡参考实现使用 Python 专家循环，不能把这里的吞吐差距解释为架构的最优 kernel 成本。没有保存训练 checkpoint、没有检查训练后读写系数的专家分化、没有全参数 FP32/BF16 逐张量梯度误差，也没有不可漂移的 FineWeb-Edu shard、完整验证集或预定的有效 batch/token 预算。因而结论仅限于 H20 上可执行、梯度有限、B/C/D 起点匹配，以及本次代理集未观察到稳定质量收益；Q2 和 EP 正确性/性能门槛仍待完成。

## 固定 Parquet 快照准备（2026-10-26）

CPU-only Slurm 作业 `2864` 在 `vllm-sgp-h20-06` 完成（`00:03:46`，退出码 `0:0`）。复用 Hugging Face Hub 下载 `HuggingFaceFW/fineweb-edu` commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 的 `sample/10BT/000_00000.parquet`，源 SHA256 `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`。PyArrow `21.0.0` 安装在 `ljj/_deps/`，没有更改共享 Python 环境。按文档 ID 哈希切分并过滤全局精确重复后扫描 21,223 行：训练 19,089 文档、20,007,425 token；验证 2,133 文档、2,183,758 token；1 个重复文本被跳过，0 个重复 ID/空文档。`ljj/data/qwen-fineweb-pinned/manifest.json` 记录 tokenizer、源 shard、bin 和 ID 清单哈希；独立 `--verify-only` 校验已通过，5 项相关测试通过。

首次 GPU 作业 `2870` 在预训练夹具测试阶段因启动脚本遗漏 `_deps` 导入路径失败，未执行训练（`FAILED 1:0`）；已修正 `scripts/run_qwen_pinned.sh` 并重新提交。完整 provenance 清单见 `evidence/h20-2026-10-26/pinned/manifest.json`，同目录的 `train.ids.gz`/`val.ids.gz` 解压后 SHA256 与 manifest 中的 ID 清单哈希一致。Parquet/token bin 本身留在被 Git 忽略的远端 `ljj/cache/` 和 `ljj/data/`。

## 固定快照 1,000 步诊断（2026-10-26）

Slurm 作业 `2871` 在 `vllm-sgp-h20-05` 的一张 NVIDIA H20-3e 上完成（`01:13:10`，退出码 `0:0`）。使用上节相同 SHA256 的训练/验证 bin、官方 Qwen3-0.6B-Base 权重和固定 Transformers fork。dense/A/B/C/D 各自从原 checkpoint 重新启动，种子 42/43/44，batch 2、序列 128、每组 1,000 步和 256,000 训练 token；前 100 步线性预热至 `1e-5`，其后余弦降至 `1e-6`，BF16 autocast、梯度裁剪 1。每 100 步评估同一批 128 个 batch，即从 2,183,758-token 的互斥验证池中固定抽取 32,768 token，**不是全量验证**。原始逐组 JSONL 均在 `evidence/h20-2026-10-26/pinned/`。所有 15 组的记录完整覆盖 step 0,100,...,1000，NLL/末步梯度范数有限；每个种子的 B/C/D BF16 起点完全一致。

| 种子 | dense 末步 | A 末步 | B 末步 | C 末步 | D 末步 | D−C | D−B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.122183 | 3.128304 | 3.128334 | 3.128224 | 3.128162 | −0.000062 | −0.000172 |
| 43 | 3.118935 | 3.124761 | 3.124935 | 3.124643 | 3.124935 | +0.000292 | ≈0 |
| 44 | 3.120660 | 3.126013 | 3.126222 | 3.126555 | 3.126608 | +0.000053 | +0.000386 |
| 均值 | 3.120593 | 3.126359 | 3.126497 | 3.126474 | 3.126568 | +0.000094 | +0.000071 |

C−B 的三个差值约为 `−0.000110、−0.000292、+0.000333`，平均 `−0.000023`。D−C 与 D−B 的方向不一致，平均变化离预设的 `−0.01 nats/token` 质量改善门槛很远。dense 在三个种子的末步 NLL 均低于 MoE 组；所有五组的末步 NLL 都比各自起点高，表明该小 token 预算与超参数下仍有明显验证退化。不能用 dense 对多流的差值单独归因于读写架构，因为 dense 参数量约为 MoE 的一半。

| 组别 | 平均训练循环秒数/1,000 步 | 峰值分配显存约 GiB |
|---|---:|---:|
| dense | 71.98 | 10.49 |
| A | 140.90 | 19.36 |
| B | 250.25 | 19.38 |
| C | 249.07 | 19.38 |
| D | 248.60 | 19.39 |

计时不含初次加载、固定数据准备、验证以及末步机制探针，不能作为完整 GPU 时或专用 kernel 性能。D 在第二个 MoE 的专家读取权重分化 RMS（三种子）约 `0.000125/0.000121/0.000126`，写回约 `0.000133/0.000128/0.000125`；C 只有写回分化，B 两者都被绑定，说明独立参数确实更新，但其幅度和质量收益没有因果等价关系。最终流中心化 RMS 约 `0.335–0.338`。仍缺完整验证池评分、训练 checkpoint、全参数 FP32/BF16 梯度对照、更多有效 token/step 及预定的 10,000 步多种子对照。**本次固定快照诊断未提供 D 的稳定质量优势，因此按预设门槛暂停专用 EP/kernel 工作；这不是多流 idea 不可行的证明。**
