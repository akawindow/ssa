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
- 下载的源码仅用于本地只读核对，不执行，不放入本仓库提交，不声称已取得作者实现的再分发许可。

当前 Python 原型参考公开代码的规则描述独立编写：移动中位数趋势、常数边界延拓、同相位季节统计、残差均值/标准差阈值、同相位残差修复。没有使用原文的随机高斯修复。NumPy 的偶数样本中位数取中间两数均值；作者辅助类尚未逐项核对，因此不声称结果一致。

本机未发现 Java/Javac/Maven，未进行与原始 Java 的数值对照。正式报告前必须补充一致性验证或明确将本方法作为有差别的替代实现。当前 SRD 相关实验只用于验证接口与训练流程。

## 任务训练与证据边界

本工程提出的组合尚未验证新颖性。固定代理 F 的梯度用于训练选择器，SRD 不在反向传播路径中。最终 H 是独立拟合的 Ridge，与代理 MLP 结构不同。第二轮增加了 3 个训练种子，但仍只有一种人工污染实现、一个数据集、小窗口子集和一个最终预测器，不能宣称具有普遍插件能力。

第二轮的共享任务特征及简单选择头借鉴你本地 TASK 工程 `base_classes.py`、`base_models.py` 的结构思路，来源：https://github.com/leizheng2005/Task-Oriented_Learning_for_Automatic_EEG_Denoising 。本工程的单变量预测模型和训练实现独立编写，没有再分发 TASK 源码或权重；TASK 本地代码采用交替训练，本轮仍使用固定代理，不能称为完整 TASK 复现。
