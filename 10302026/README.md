# ART + SCP 吸气式巡航轨迹优化

**项目维护文档，更新于 2026-10-30。** 本文件统一记录项目目标、架构、运行方式、验证结论、已知问题和后续任务。修改代码或实验设置时同步更新本文件。

## 目标与当前状态

复现 *Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous* 的方法，将对象替换为吸气式巡航飞行器。

```text
巡航物理模型 → 成对专家轨迹 → [RTG, CTG, state, action] → ART → SCP → 独立积分验证
```

采用 CruiseTrjOpt 的巡航物理模型与独立积分验证，结合原项目的 ART/SCP 方法主线。活动代码集中于 `cruise/`，车辆与默认任务配置集中于 [configs/cruise.yaml](configs/cruise.yaml)。

已完成代码合并、旧实现清理、统一数据格式、两种 ART 推理方式和端到端流程验证。已有 4 个随机任务、8 条成对轨迹和一个小规模验证模型。下一阶段开展正式数据生成、论文配置训练及多任务统计比较。

## 架构与文件职责

| 文件 | 职责 |
| --- | --- |
| `cruise/config.py` | 车辆、任务、物理限制、数值设置与缩放 |
| `cruise/dynamics.py` | NumPy/CasADi 共用动力学、RK4 转移及导数 |
| `cruise/geometry.py` | 球面航迹与禁飞区距离 |
| `cruise/optimization.py` | IPOPT 专家解、SCP、迭代记录 |
| `cruise/validation.py` | DOP853 独立积分、终端与路径约束复核 |
| `cruise/dataset.py` | 成对样本、RTG/CTG、按任务划分数据 |
| `cruise/model.py` | 四模态因果 ART 与任务条件 token |
| `cruise/learning.py` | Teacher forcing、验证集选择、检查点 |
| `cruise/inference.py` | Transformer-only 和 dynamics-in-the-loop 推理 |
| `cruise/evaluation.py` | 留出任务上的预测与热启动比较 |
| `cruise/plotting.py`、`cruise/cli.py` | 绘图与统一命令入口 |
| `tests/` | 动力学、标签、因果性、独立积分和求解测试 |
| `data/`、`checkpoints/`、`results/` | 生成数据、模型和实验结果 |
| `references/` | 原论文、原项目压缩包、参考源码归档和一份原始航天器数据 |
| `doc/*.json` | 清理核对清单 |

## 物理模型与统一约定

- 七维状态依次为 `[h, longitude, latitude, V, gamma, heading, mass]`；三维控制依次为 `[alpha, bank, throttle]`。内部使用 SI 单位，角度为弧度，航向从北顺时针计量。
- 采用旋转球形地球、US76 巡航层、线性升力系数与二次阻力极线。推力为 `T = throttle * CT * q * S`，质量变化为 `dm/dt = -T / (Isp * g0)`。当前气动和发动机参数来自 CruiseTrjOpt 的概念飞行器。
- 每个任务固定时域；不同任务可以具有不同飞行时间。控制采用零阶保持，状态形状为 `(N+1, 7)`，控制形状为 `(N, 3)`。零阶保持的控制切换由独立积分逐区间处理。
- 初始七维状态固定；终端固定高度、经纬度、速度和航迹倾角，终端航向与质量自由。
- 专家解、SCP 和 ART 动力学推理共用 RK4 离散转移。独立验证从原始初态重放控制，采用 DOP853 和约 1 秒检查网格，核对动力学节点、终点、状态/控制范围、动压、热流、过载和禁飞区。
- RTG 为负的剩余耗油量：`R[k] = -(mass[k] - mass[N])`，统一以 1000 kg 缩放。推理递推为 `R[k+1] = R[k] + fuel[k]`。
- CTG 为未来进入任一禁飞区的时间步数，同一步多个禁飞区违规计一次。训练和推理均以区间数缩放，推理递推为 `C[k+1] = C[k] - violation[k]`。

默认任务约 4000 km、1920 s，100 个控制区间；初末高度 30 km、初末速度 2110 m/s、初始质量 20 t，包含三个禁飞区。RK4 最大步长为 2 s。

## 与论文的对应和迁移选择

| 论文方法 | 当前实现 |
| --- | --- |
| 式 3–6 的 `[R,C,x,u]` 表示 | 四种线性编码、时间嵌入和因果 attention |
| 式 8 的 teacher forcing 与平方误差 | 状态 token 预测当前动作，动作 token 预测下一状态；联合训练两个预测头 |
| Algorithm 1 | `rollout(..., mode="transformer")` 通过状态预测头推进 |
| Algorithm 2 | `rollout(..., mode="dynamics")` 通过共用 RK4 动力学推进 |
| 式 20 的负剩余控制代价 | 替换为负剩余燃油质量 |
| 式 21 的非凸避障 CTG | 替换为进入任一禁飞区的时间步计数 |
| 放松问题与全约束问题成对生成数据 | 禁飞区放松的非线性专家解与全约束非线性专家解 |
| ART 轨迹作为 SCP 初值 | 两种推理均与几何初值、放松解初值进行相同 SCP 比较 |
| 表 4 的默认超参数 | 6 层、6 头、384 维、K=100、ReLU、dropout 0.1、批量 4、累积 8、lr 3e-5、梯度裁剪 1 |

巡航任务包含非线性气动、推进、质量变化和地球自转，SCP 同时线性化动力学与路径约束，并加入虚控制和约束松弛。在线放松解用于提供 RTG 参考值；局部 NLP 的全局最优性需要额外证据。

为了区分变化的目标、时域和禁飞区，在四模态序列前增加一个任务条件 token，包含五维目标、飞行时间和带存在掩码的禁飞区特征。该 token 属于对象迁移的扩展。

同一任务的成对轨迹整体划入同一数据分区。10% 任务留作测试，其余任务中再留 10% 作验证，小数据集按完整任务取整。检查点保存任务划分和物理配置指纹，评估沿用保存的划分。

## 环境

项目目录为 `D:\Experiments\TransformerTrajectoryOptimization`，已有 Conda 环境为 `hypersonic-tto`。

```powershell
conda activate hypersonic-tto
python -m pip install -r requirements.txt
python -m pytest tests -q
```

需要新建环境时执行 `conda env create -f environment.yml`。核心依赖集中声明于 `pyproject.toml`：NumPy、SciPy、CasADi、CVXPY/CLARABEL、PyTorch、PyYAML。测试与绘图作为可选依赖声明，`requirements.txt` 安装这些选项。

已验证环境：Python 3.11.17、NumPy 2.4.6、SciPy 1.17.1、CasADi 3.8.2、CVXPY 1.9.3、CLARABEL 0.11.1、PyTorch 2.10.0+cu130；GPU 为 RTX 4060 Laptop。完整版本记录见 [results/validation.json](results/validation.json)。

## 运行方式

在项目根目录运行。`--config` 和 `--intervals` 属于 `solve`、`generate` 子命令，写在子命令后；训练和评估读取数据或检查点中保存的配置。

```powershell
# 全约束专家解、禁飞区放松解、几何初值 SCP
python -m cruise solve --method expert --output results/expert.npz
python -m cruise solve --method relaxed --output results/relaxed.npz
python -m cruise solve --method scp --output results/scp.npz

# 指定配置与离散区间数
python -m cruise solve --config configs/cruise.yaml --intervals 100 --method expert

# 状态、控制与约束曲线
python -m cruise plot results/expert.npz --output results/expert.png

# 数据生成、训练和留出任务评估
python -m cruise generate --count 100 --workers 4 --output data/cruise
python -m cruise train --data data/cruise --output checkpoints/art.pt
python -m cruise evaluate --data data/cruise --checkpoint checkpoints/art.pt --output results/benchmark.json
```

`generate --count` 统计独立任务，每个任务存两条轨迹。每条轨迹通过独立积分及对应约束验收后才写入 `mission-*.npz`。生成器记录成功与失败，可复用已有同配置样本。随机采样筛选端点条件，全程可行性通过求解和积分验证确定。

小规模流程验证数据、检查点和评估结果已经保存，可直接复查：

```powershell
python -m cruise evaluate --data data/cruise_smoke --checkpoint checkpoints/smoke_art.pt --output results/smoke_benchmark.json --limit 1
```

## 求解状态与计时

SCP 根据实际非线性罚函数和局部模型的下降量接受或拒绝步骤。默认最多 40 次成本优化子问题；末次迭代接近可行时，再进行最多 3 次带近端正则的凸可行性恢复，全部计入迭代数与耗时。

- `success`：输出通过物理可行性验收；SCP 同时检查离散动力学缺陷。
- `converged`：满足已设置的优化停止条件。
- `feasible_iteration_limit`：达到成本迭代上限，通过恢复阶段取得可行输出，保留 `converged=false`。
- `runtime`：函数内图构建、计算和独立验证的总时间。ART 的生成循环耗时另存于 `validation.generation_runtime`。
- `total_runtime`：评估时加上相应的初始化或在线放松求解时间。

## 已完成的验证

最近一次自动化测试：**9 passed**。覆盖旋转地球动力学的独立笛卡尔力平衡、US76 数值点、离散转移导数、RTG/CTG、分组划分、因果 attention、虚假终点检测、SCP 独立积分和推理 RTG 递推。本次维护另以短航程实际执行 `expert`、`relaxed`、`scp` 三种 CLI 求解，核对配置、区间数和独立积分。

默认 100 区间巡航场景的结果：

| 方法 | 物理复核 | 耗油 kg | 实际终端位置误差 m | 最小禁飞区余量 m | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| IPOPT 专家解 | 通过 | 4417.392 | 0.051378 | 196.363 | `Solve_Succeeded` |
| 几何初值 + SCP | 通过 | 4777.173 | 0.008214 | 195.357 | `feasible_iteration_limit` |

该几何初值 SCP 执行 40 次成本子问题和 1 次可行性恢复，耗油比专家解高约 8.14%，成本结果仍需通过改进初值和收敛设置改善。数据见 [专家解](results/cruise_expert.json)、[SCP 记录](results/cruise_scp.json) 与对应 `.npz`；曲线见 [巡航状态和控制](results/cruise_expert.png)。

小规模验证包含 4 个任务、8 条轨迹。任务 100、101 用于训练，103 用于验证，104 用于测试。放松轨迹初始 CTG 分别为 15、5、7、8，全约束轨迹均为 0。数据为 `data/cruise_smoke/mission-*.npz`，候选任务记录为 `data/cruise_smoke/generation.jsonl`。

验证模型采用 2 层、4 头、64 维、60 步上下文，训练 100 个 epoch、批量 4、累积 1、学习率 1e-3。最佳验证 epoch 为 97，联合归一化 MSE 为 0.014717，检查点为 `checkpoints/smoke_art.pt`。

留出任务 104 的单次结果：

| 方法 | 物理复核 | SCP 迭代数 | 耗油 kg | 实际终端位置误差 m | 总时间 s |
| --- | --- | ---: | ---: | ---: | ---: |
| 几何初值 + SCP | 通过 | 28 | 4562.808 | 0.168907 | 40.848 |
| 放松解 + SCP | 通过 | 28 | 4469.899 | 0.108700 | 119.981 |
| ART Transformer-only + SCP | 通过 | 30 | 4449.119 | 0.095238 | 134.939 |
| ART dynamics-in-loop + SCP | 通过 | 30 | 4470.598 | 0.182184 | 122.263 |

该任务的离线专家耗油为 4448.462 kg。两种纯 ART 预测经独立积分后的终端偏差分别约 211.1 km、227.8 km。经过 SCP 后均通过验收。当前结果用于核对流程及初值影响，模型预测质量和速度优势继续通过扩大样本与测试任务评估。

完整指标与迭代历史见 [results/smoke_benchmark.json](results/smoke_benchmark.json)，所有专家、预测与优化状态/控制数组见 [results/smoke_benchmark-104.npz](results/smoke_benchmark-104.npz)。

## 清理记录与恢复

首次整合已清理 136 个旧文件，修正航向曲率/科氏项符号、RTG 符号和缩放、终端约束、燃油目标及基于独立积分的可行性判断。移除了重复的单/多禁飞区模型、PPO、可微 SQP、临时脚本、旧权重与旧结果，依赖统一到当前包。逐文件 Git blob 清单见 [doc/cleanup_manifest.json](doc/cleanup_manifest.json)。

本次维护进一步完成：

- 参考仓库 28 个源码文件与源码归档逐一核对，完整 Git 历史保存并验证为 `references/paper-history.bundle`。
- 14 个重复数据/权重文件逐一核对 SHA-256，保留 `references/paper_data/` 中的一份数据。清理旧参考 checkout 共 78 个文件，释放约 4.43 GiB；核对清单见 [doc/reference_cleanup.json](doc/reference_cleanup.json)。
- 移除动力学中未使用的中间采样构造和约束分支。
- 配置参数限定到实际使用它们的子命令；以 `solve --method relaxed` 明确选择放松问题，清理无效参数组合。
- 原分散的合并说明、验证说明与资料说明并入本文件。

| 资料或版本 | 恢复位置 |
| --- | --- |
| 修改前的主项目、项目数据、权重和报告 | Git 标签 `archive/pre-consolidation-2026-10-30`，提交 `8d61489` |
| 首次整合与本次维护前版本 | 提交 `6bb90d1`；标签 `archive/pre-maintenance-2026-10-30` |
| 用户提供的 CruiseTrjOpt | `references/CruiseTrjOpt.zip` |
| 论文 | `references/Transformers_for_Trajectory_Optimization_with_Application_to_Spacecraft_Rendezvous.pdf` |
| 论文配套源码及许可证 | `references/art-paper-source.zip` |
| 论文参考仓库完整历史 | `references/paper-history.bundle` |
| 原始航天器数据的一份副本 | `references/paper_data/`，由 Git 忽略 |

```powershell
git log --oneline --all
git diff archive/pre-consolidation-2026-10-30..HEAD --stat
# 按需提取一个旧文件
git show archive/pre-consolidation-2026-10-30:scp_solver/scp_pipeline.py | Set-Content -Encoding utf8 old_scp_pipeline.py
```

## 后续任务与维护约定

后续任务按顺序推进：

1. 扩大成对专家数据，统计采样分布、拒绝率和可行性。
2. 按论文默认配置正式训练，保持成对任务的训练/验证/测试划分。
3. 在多个独立测试任务上比较预测误差、SCP 迭代、耗油与包含初始化的总时间，并测试 RTG/CTG 条件变化。
4. 改善当前几何初值 SCP 的耗油差距；按需要替换概念飞行器的气动和发动机参数。

后续修改时维护以下内容：

- 改动模型、单位、标签或离散化时，同步更新科学约定与数据/检查点版本信息。
- 改动接口时，同步更新实际可执行的命令和文件路径。
- 开展实验后，将设置、原始结果路径、验证结论和适用范围写入本文件；数值依据保存在机器可读结果文件中。
- 清理文件前核对 Git 存档或保留副本，清理后检查导入、命令和相关测试。
- 完成阶段工作后更新本文件日期、当前状态和后续任务，并提交本地 Git。

所有工作保留在本机。
