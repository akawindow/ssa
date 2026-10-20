# 数据、算法来源与验证边界

## 数据

Power Consumption of Tetouan City：Salam, A. & El Hibaoui, A. (2018), UCI。

- https://doi.org/10.24432/C5B034
- https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
- 许可 CC BY 4.0；下载文件和具体 SHA256 由下载脚本保存于被 Git 忽略的 `data/downloads/manifest.json`。
- 原始 CSV，不使用 SRD 论文为了规模实验拼接扩充的版本。采用 Zone 1 单变量。

## SSA

按基本轨迹矩阵嵌入、SVD、分量对角平均独立实现。

- https://research.atmos.ucla.edu/tcd/ssa/guide/andy/ssatheory4.html
- 残差组用于保留未显式计算的成分；固定 SSA 基线与任务选择器共享同一分解。

## 季节修复原型

- 论文：Cleaning Time Series under Seasonal and Trend Constraints，https://doi.org/10.1145/3769773
- 作者工程：https://github.com/czjthss/SeasonalClean
- 核对入口：`code/src/main/java/Algorithm/SRD.java`、`Algorithm/util/Decomposition.java`。
- 最初两份源码用于本地核对；后续按用户授权下载并构建完整作者工程，在本地执行基线。作者源码、JAR和数据不放入本仓库提交，不声称已取得再分发许可。

当前 Python 原型参考公开代码的规则描述独立编写：移动中位数趋势、常数边界延拓、同相位季节统计、残差均值/标准差阈值、同相位残差修复。没有使用原文的随机高斯修复。NumPy 的偶数样本中位数取中间两数均值；作者辅助类尚未逐项核对，因此不声称结果一致。

初期未在系统路径发现Java；后续在IDE安装目录发现完整JDK21和Maven3.9.9并按项目复用。新增Java后端固定作者提交 `84486b6c15e02f17ac32c80b827afb17e09802b9`，直接调用未修改的作者JAR；本项目自己的适配器源码位于 `java/SRDBridge.java`。

作者smoke与Python批量接口的2000个输出逐点完全一致，作者原始源码受版本管理文件无修改；JAR和适配器均保存哈希。旧Python原型与作者输出存在实测差异，保持原型标签，不视为已完成移植校准。当前已跑通作者代码和小型基线，不代表完整复现论文所有表格，见 `docs/JAVA_SRD_RESULTS.md`。

## 任务训练与证据边界

本工程提出的组合尚未验证新颖性。旧版本固定代理F监督选择器；第四轮新增TASK式交替更新，选择器阶段只更新权重头，代理阶段同时更新共享任务特征。两类模式都不通过SRD反向传播。最终H是独立拟合的Ridge，与代理MLP结构不同。虽已增加3个训练种子，仍只有一种人工污染实现、一个数据集、小窗口子集和一个最终预测器，不能宣称具有普遍插件能力。

共享任务特征、简单选择头与第四轮协同机制借鉴你本地TASK工程 `base_classes.py`、`base_models.py` 的结构和训练顺序，来源：https://github.com/leizheng2005/Task-Oriented_Learning_for_Automatic_EEG_Denoising 。本工程的单变量预测模型和训练实现独立编写，没有再分发TASK源码或权重。第四轮迁移了交替优化核心，但数据、网络、任务损失及部分稳定化设置不同，不能称为完整EEG TASK复现。详见 `docs/COLLABORATIVE_RESULTS.md`。

第七轮重新核对本机TASK实现，新增完整SSA成分、零偏置/max-norm选择头及等预算原始代理、常数缩放和昨日持久性对照。原始工程只读，文件哈希记录在执行元数据中；新配置不改变旧默认值。TASK式头仅迁移选择头的两项设置，并未采用整个EEGNet。当前预测改善未超过简单缩放，不能称为已经证明的去噪效果。见 `docs/TASK_ALIGNMENT_AUDIT.md` 和 `docs/TASK_ALIGNMENT_RESULTS.md`。
