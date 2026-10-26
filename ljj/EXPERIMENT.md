# 多流 MoE：可执行实验规格（v1）

本文件是 A/B/C/D 实验的权威规格。README 给出动机和理论边界；旧 `docs/EXPERIMENT.md` 是初稿，若有冲突以本文件为准。当前单卡参考代码不等于完整配置或专家并行。优先验证下面的 Qwen warm-start 轨道，再决定是否投入从零训练和专用 kernel。

## 1. 问题与预注册比较

给定相同数据、token 预算和训练日程，比较 D（专家独立读写）对 C（仅独立写回）、C 对 B（共享读写）的验证 NLL、训练 GPU 时和显存。A（普通单流 MoE）是成本及退化参照，不把 A/B 的差异单独解释为专家差异化收益。先验证结构，再投入专用派发/回传优化。分别报告架构收益和同一 D 权重、同一路由下的 infra 加速；不能混为一个数字。

## 2. 完整模型语义

张量形状 `X: [batch, time, streams, hidden]`；A 有一流，B/C/D 有四流。token/位置嵌入被复制到各流，最后对流取均值、做最终归一化和共享的词表投影。所有注意力均为严格因果注意力；输入输出嵌入绑定。

每个 block 先取 `h=mean_stream(X)`，执行 `Y=X+1_stream * Attention(LN_attn(h))`。奇数编号 block 再做 `Y+1_stream * SwiGLU(LN_dense(mean_stream(Y)))`；偶数编号 block 是 MoE：

```math
Y^+=R(Y)Y+b_s(Y)E_s(a_s(Y)Y)+\sum_{j\in Top2(Y)}g_j(Y)b_j(Y)E_j(a_j(Y)Y).
```

此处编号从 1 开始。共享专家和 Top-2 专家读取同一个 **MoE 入口旧状态** `Y`；共享专家始终在源设备执行，不等待路由专家。`E_j` 内的 LayerNorm/RMSNorm 与 SwiGLU 顺序必须在各组相同。路由器读取 `c=LN_router(mean_stream(Y))`；先取两项最高 logits，再对这两项做 softmax 得 `g`。不丢 token；辅助均衡损失对四组使用相同公式、系数 0.01，并单独记录 NLL（不把辅助项并入验证指标）。

首版的动态系数由源设备上相同的上下文 `c` 产生：`a_j(Y)=softmax_stream(W^a_j c+u^a_j)`；`b_j(Y)=2 sigmoid(W^b_j c+u^b_j)`。共享专家也有自己的专家编号 0。B 的所有专家共用一对生成器；C 只拆分写回生成器；D 两者都按专家拆分。**相同结构中各组生成器输入、非线性及计算位置不变**，只解除参数绑定。只对实际选中专家生成系数；源端保存/重算反向所需上下文、系数及路由。

残差矩阵 `R(Y)=0.95 I+0.05 Sinkhorn(S(c))`，其中 `S(c)` 是逐 token 的 `4x4` 线性投影 logits；用同一 FP32、20 轮交替行列归一化实现。A 为 `R=I`。记录每层最差非负性、行列和误差，超过 `1e-4` 视为正确性失败；低精度乘法的均值误差另外报告。输入相关矩阵的 Jacobian 不受固定矩阵奇异值界约束。注意力/稠密层不额外混合流，流差异可持续到下一个 MoE。

生成器权重和残差 logits 的线性权重初始化为零。读取偏置初始化为零；写回偏置令 `b=(1.03,0.99,0.99,0.99)`（均值 1）。C/D 的独立行逐行复制 B 的初值，并复制所有其他公共参数；逐层比较 B/C/D 初始 logits、状态和路由。A 的公共参数按名字复制：初始化时 B 的流均值应退化到 A；训练后不要求两者相等。记录各组总参数、每 token 激活参数、实际 FLOPs，避免把新增容量误称为免费。

## 3. 正确性门槛

1. CPU FP64 小张量固定路由，对 `Y`、生成器权重/偏置、专家权重、门控和 `S(c)` 做差分/gradcheck；共享专家与空专家路径均覆盖。TopK 索引不连续处不做可导断言。
2. 检查 Sinkhorn 行列和及流均值，A/B 初始退化、B/C/D 初始前向相同；写回后各流确实不全相同；D 的读取生成器在后续 MoE 接受非零梯度。
3. 单 H20 上 FP32 对照 BF16 的前向/反向（给出逐张量绝对和相对误差）；固定数据和种子跑四组各 32 步，无 NaN/Inf。任何不满足者不得进入性能测试。
4. 多 GPU：固定 token-to-expert 列表，EP 与单卡参考的前向、输入/生成器/专家梯度、优化器一步后参数对照；包含同 token 两专家同目的设备、空专家和不均匀负载。此阶段尚未实现，不能写成已通过。

## 4. 数据、配置和预算

完整训练使用 FineWeb-Edu `sample-10BT` 与 TinyLlama tokenizer，固定**不可漂移的**数据/分词器提交号、SHA256、文档标识、训练顺序及文档级互斥验证切分；记录重复检测规则和约 200 万验证 token 的名单。绝不从验证文档抽训练窗口。所有组读取同一缓存 token 序列、同一顺序及评估窗口。短时工程冒烟可使用 nanoGPT Tiny Shakespeare，但其结果不用于研究结论。

主配置：12 层、宽 512、8 头、序列 1024、词表 32,000（以 tokenizer 实际词表与 padding 为准）、6 个 MoE 层，各层共享专家 1＋路由专家 8、Top-2，专家 SwiGLU 中间宽 768，稠密宽 1,536。BF16；全局有效 token/step 32,768；AdamW `lr=3e-4, betas=(0.9,0.95), eps=1e-8, wd=0.1`，梯度裁剪 1。EP=4 或 8、TP=1 时由数据并行和梯度累积保持有效 token 数。所有通用融合、激活重算、精度、优化器和调度设置一致。

工程排错轮：每组 1,000 步、种子 42，按 **1,000 步独立调度**（前 100 步预热，余弦至 `3e-5`）。这是诊断，不是性能结论。扩展轮若资源批准，**从头重训**每组 10,000 步、配对种子 42/43/44，前 1,000 步预热、余弦至 `3e-5`；不得续接诊断轮的 1,000 步 checkpoint。四组总 3.93216e9 token。每 100 步固定窗口评估、记录 checkpoint 和随机数状态，报告完整曲线。

## 5. 指标与决策

结构主指标：配对种子的验证 NLL 对累计训练 GPU 时曲线、达到同一 NLL 的成本；固定 token 步数下报告 D-C 和 C-B 差值。报告每种子全部曲线、均值/范围，不用三种子声称统计显著。预设探索性门槛：D 相对 C、B 的三个配对差值方向一致、末步平均至少改善 `0.01 nats/token`，且在相同 GPU 时不落后；若未达到，记录负结果并暂停专用 kernel 投入。指标也包括有效 token/s、端到端 step 时间、训练与评估时间分开、专家负载、流间差异/相似度、参数与 FLOPs、峰值显存和故障重试。

系统指标只比较 **同一 D checkpoint、同一 token/路由/数学语义** 下的成熟 dispatcher 与专用融合；基线启用所有适用的通用优化。30 步预热、100 步计时、3 次重复，报告中位数/范围和完整训练步吞吐，目标至少 10%；同时分别记录 `read+pack`、all-to-all、专家计算、`return+write` 的 CUDA 时间线、HBM 中间张量和实际收发字节。精度和梯度门槛不通过则性能数字无效。

## 6. infra 路径与限制

先测 H20 卡间拓扑、NCCL/驱动/CUDA/编译器及 dispatcher 兼容性。优先复用 Megatron Core 的 MoE/EP dispatcher 或经 H20 验证的 DeepEP 版本；不得仅凭分区名推断 NVLink、RDMA 或可用的 kernel。源设备生成所选专家的 `a_jY`，保持专家身份派发；目的专家算 `z_j`，回传逐专家结果，源端分别累积 `g_j b_j z_j`。共享专家可与路由分支重叠，但必须读取旧状态。若复用库在 combine 时提前聚合，必须绕过该聚合点，不能改变写回数学语义。

依次测未融合、仅读取/打包融合、仅回传/写回融合、两者融合，并保持同一路由与完整反向。Top-2、每设备一个专家时，同一 token 到同一设备的专家数至多 1，公共基底传输没有可预设的收益；先收集放置/秩与成本分布，再决定是否做单独微基准。跨机、强制低秩和更大 TopK 是独立实验，不并入主结论。

## 7. Qwen3-0.6B-Base warm-start 轨道（优先）

使用官方 `Qwen/Qwen3-0.6B-Base` 的权重和 tokenizer，锁定 revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。框架从 `https://github.com/inaniloquentee/transformers` 的 `ljj/qwen3-0.6b-validation` 分支获取，严格校验上游 `v5.19.0` 提交 `c4c4605f961877852b89b97478536bb3a26b6fd1`，运行时设置 `PYTHONPATH` 到该 fork 的 `src`，不把整个框架复制进本仓库。复用 Qwen3 attention、RMSNorm、RoPE、SDPA、MLP 和 PyTorch autograd；已有 nanoGPT fork 仅供从零训练的小参考模型。把从 0 开始的奇数层 MLP 换成 1 个共享＋4 个 Top-2 路由专家，所有专家拷贝原 MLP 和 post-attention RMSNorm，输出投影乘 0.5；因为 Top-2 权重和为 1，初始共享＋路由更新仍等于原 MLP。每层注意力和未替换 MLP 在流均值上执行，结果广播回各流；MoE 之间保留独立残差流。先检查原 dense checkpoint 与五组在固定 token 上的初始 logits，及 A/B/C/D 的路由、流差异、第二个 MoE 的读取梯度。报告最大绝对误差、相对误差及状态 RMS，不能只比较最终损失。

五组固定：原始 dense Qwen 是保留预训练模型的强锚点；A 是同专家数、同 checkpoint 提升的普通单流 MoE；B/C/D 只改变共享/独立读写参数绑定。dense 的参数/FLOPs 与 A 不相等，**不得把 dense 对多流的差异解释为隔离的架构效应**。D-C 与 C-B 才是主要结构对照。每组记录参数量、实际 token/s、峰值显存和累计训练 GPU 时。初始路由器和所有复制参数使用同一种子，数据窗口与验证窗口逐组完全相同。dense 路径直接计算每个输入 token 对其下一个 token 的交叉熵，不使用模型内置的额外 shift。

阶段 Q0：先在 `ljj/` 执行 `bash scripts/bootstrap_transformers.sh`、`source scripts/use_transformers_fork.sh`，再运行 `python -m unittest -v tests.test_qwen`、`python -m experiments.check_qwen_checkpoint --model cache/qwen3-0.6b-base --data data/qwen-fineweb-proxy`，并记录打印的 `transformers_source`。小模型 CPU 反向及真实 checkpoint 前向等价；用 `python -m experiments.check_qwen_bf16` 记录真实 checkpoint 的 FP32/BF16 NLL、logits 差及关键读写梯度。B/C/D 在同一固定输入上的 BF16 初始 NLL 必须相同（浮点舍入容差 `1e-4`），否则不能解释后续消融；dense/A 初始 BF16 误差单独报告。单样本预检不推断效果。阶段 Q1：单 H20、BF16、同一机器依次跑 `dense,A,B,C,D` 各 32 步，batch=1、seq=128、学习率 `1e-5` 余弦衰减至 10%、AdamW `betas=(0.9,0.95),wd=0.1`、梯度裁剪 1；每组 8 个固定验证窗口，须再次检查 B/C/D 的 step 0 NLL 一致。`python -m experiments.prepare_qwen --max-rows 1000` 从 FineWeb-Edu dataset viewer 构造的 939,516/123,095 token 文档级拆分**只是易变的工程代理集**，即使记录了数据 SHA256 和观察到的 revision，也不是锁定 parquet 快照；它可检测崩溃、初始等价和成本，不可作发表或可重复的质量结论。Q1 至少记录每步梯度有限性、step 0/16/32 的 NLL 和峰值显存。

阶段 Q2（需要 H20 配额和固定语料快照）：选取并记录不可漂移的 FineWeb-Edu shard commit、文件 SHA256、文档 ID、去重规则、Qwen tokenizer revision，按文档划分互斥且固定不少于约 200 万验证 token。先做 1,000 步诊断，不从代理集 checkpoint 续训；有资源后按 10,000 步计划从同一个 base checkpoint 重新开始，五组配对种子 42/43/44，预注册相同窗口、batch、seq、有效 token 数和 schedule，并发布原始曲线及 GPU 时。比较 D-C、C-B 三种子的逐对差值、末步方向和同 GPU 时成本；达到第 5 节探索性门槛才投入 EP/融合。若 Q2 无法执行，只报告 Q0/Q1 正确性与成本，结论为**有效性未验证**。

固定快照 1,000 步诊断的实际预设：FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 的 `sample/10BT/000_00000.parquet`，源 SHA256 `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`；按文档 ID 哈希作 90/10 切分、跳过全局精确重复文本，生成至少 2,000 万训练和 200 万验证 token。五组各用种子 42/43/44、batch 2、seq 128、1,000 步（每组 256,000 训练 token）、BF16、AdamW `lr=1e-5`，100 步线性预热后余弦降至 10%；每 100 步在相同的 128 个 batch（32,768 token）上评估，记录训练循环秒数、显存和末步流/读写参数差异。验证 corpus 大于评估窗口，不可把 32,768 token 采样称为全量验证；这仍是诊断，不替代预设的 10,000 步、多种子、完整固定验证实验。

## 8. 当前交付状态

`ljj/multistream_moe/multistream.py` 与 `ljj/experiments/train.py` 是单卡、无 EP 的从零训练参考和冒烟入口；`ljj/multistream_moe/qwen_reference.py` 与 `ljj/experiments/train_qwen.py` 是优先的 Qwen 路径。最小有效对照至少需要两层 MoE（当前隔层 MoE 即至少四层）：第一层的非均匀写回生成流差异，第二层的独立读取才能使用它；只有一层 MoE 的 C/D 比较是无效检验。H20 上的代理集 Q1 和额外的 1,000 步多种子诊断已完成，见 `ljj/RESULTS.md`；它们不满足固定快照 Q2，不能称为多种子质量结论。完整数据训练、EP 对照、自定义融合仍**未完成**。每次 PR 必须附实际命令、提交号、环境清单、原始日志及失败项，不以本规格代替实验结果。
