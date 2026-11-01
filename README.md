# ART-PPO-based-on-ART-SCP：巡航轨迹优化

本仓库包含两套巡航轨迹优化代码及本地 PyCharm 入口。

## 内容
- CruiseTrjOpt/CruiseTrjOpt：LGL 初始化、IPOPT 与连续多重打靶、独立积分验收。
- 10302026/cruise：专家数据生成、ART 训练和推理、SCP 优化、独立验证。
- 10302026/cruise/reinforcement.py：实验性 PPO 残差控制，当前短训练未通过物理验收。

## 本地运行
Python 3.11。当前机器使用项目 .venv（基于 Anaconda O^O）。新机器请自行创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".\10302026[test,plot]" pandas h5py
.\.venv\Scripts\python run_cruisetrjopt.py
.\.venv\Scripts\python run_art_scp.py
.\.venv\Scripts\python run_rl.py --updates 50 --extra-zones 2 --seed 42
```

run_art_scp.py 当前执行 IPOPT 专家求解，并不会自动训练 ART。
plot_comparison.py 需要结果目录中对应的三组数据；它展示不同场景结果，不代表公平算法对比。

## 结果与限制
results 中保存轨迹、验证 JSON 和图片。当前 PPO 数据仅为 1 轮、10 区间短测试，不满足终点、热流及禁飞区约束，不能作为成功优化结果。
专家解和附带 SCP 结果满足对应场景的独立积分验收，不能据此宣称全局最优。
参考资料、训练数据、权重、虚拟环境及缓存默认不入 Git。ART 附带权重和小数据集留在本机；新环境需要另行提供数据或生成并训练。

## Git 工作方式
修改前创建分支；只暂存相关源码、配置、测试和说明。提交前查看 git diff --cached。
正式实验采用独立输出目录，记录提交号、配置、种子和验收结论。不要提交令牌、密码或 .env。
