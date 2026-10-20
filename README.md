# SSA 任务驱动去噪与季节清洗：初步实验工程

本工程独立于原 EEG 复现目录。首先研究任务训练的 SSA 选择器能否帮助参与协同训练的MLP，独立预测器用于进一步检验迁移能力。当前版本是用于排错的小规模探索，不是论文结论。

**持续研究约定：先验证“选择器＋MLP”是否超过同结构、同初始化、同代理训练预算的“原始输入＋MLP”；独立Ridge是次级迁移检验。** Ridge未改善不能被写成任务驱动训练未实现，代理多训练后的改善也不能直接算作去噪收益。继续工作前先读 [研究决策](docs/RESEARCH_DECISIONS.md) 与本工程 `AGENTS.md`。

**最新：同MLP主比较已完成。** [第九轮记录](docs/MLP_PRIMARY_RESULTS.md)重新训练三个共同预训练状态和15个匹配分支，200拟合/51选检查点/54评价；本轮不运行独立Ridge。任务选择器平均RMSE2148.83，同预算原始MLP2127.25，暂未显示稳定增量，昨日基线1647.79仍最好。36项检查和18个保存模型的预测重载核验通过。评价段以前已被探索，本轮未用它选检查点。

**当前先只看原始数据上的 SSA：不加人工污染、不接 SRD。** 已完成六次任务权重训练、19组固定诊断和24项检查。固定前六成分与两种任务加权的平均预测均未超过不处理，保留更多成分后整体接近不处理。见 [SSA 单独实验记录](docs/SSA_ONLY_RESULTS.md)。新入口为 `ssa_static`（固定诊断）和 `ssa_only`（六次训练加固定诊断）。本轮统一采用 float64 直接 Ridge 求解，避免完整重构核验中发现的近似求解误差；历史分数保留其原设置。

作者 Java SRD 后端已在第三轮构建并逐点核验，仅为历史实验和后续需要保留，见 [原版Java基线记录](docs/JAVA_SRD_RESULTS.md)。旧 `pilot` 和 `round02` 保留 Python 原型设置用于历史复查；`java_pilot`、`java_round` 保留作者 Java 设置。

**已补充训练数据的 SSA 结构诊断。** 36 个训练窗口中有 23 个的 C6/C7 满足候选配对条件，前六截断会将它们拆开；残差也有规则的较短周期变化。分组可疑之处已找到，但本轮没有新增预测成绩，不能称为已经修好。见 [结构诊断记录与图形](docs/SSA_STRUCTURE_RESULTS.md)，入口为 `ssa_structure`，28项检查通过。

**已回到本地TASK代码逐项核对，并完成单变量设计对照。** [代码核对](docs/TASK_ALIGNMENT_AUDIT.md)明确了前六聚合、偏置4等额外假设；[第七轮结果](docs/TASK_ALIGNMENT_RESULTS.md)使用251训练/54验证窗口，完成12次匹配协同训练、等预算原始代理和尺度对照。TASK式选择头使分数改善，但未超过常数缩放，且仍落后于昨日持久性基线。

**独立迁移诊断：尺度控制与训练段内Ridge选参已完成。** [第八轮记录](docs/SCALE_FAIR_RESULTS.md)冻结已有输出，对各历史位置标准化，按三个时间前缀内折选择alpha。校准原始输入RMSE1835.14，常数缩放与其一致；四种任务设置均值1858.23–1877.94，均未超过原始输入，昨日基线1647.79仍最好。34项检查通过。这是已有表示的读出诊断，源神经检查点此前用过当前验证集，不能当成端到端嵌套验证或新独立测试。

**协同训练已实现并完成匹配对照。** 第四轮同时比较冻结代理与TASK式交替更新、共享特征更新，共12次验证运行、19项检查。代理自身改善，但独立预测没有稳定增量，见 [协同实验记录](docs/COLLABORATIVE_RESULTS.md)。`java_pilot`仍保留旧冻结设置；`collaborative`入口会运行新的两组匹配对照。

已完成的首次运行、数值、适用范围和下一步见 [初步实验记录](docs/INITIAL_RESULTS.md)。

第二轮已补齐无额外污染对照、TASK 式共享特征与匹配随机特征对照，完成 18 次验证集运行及训练前后诊断。结果与下一步见 [第二轮实验记录](docs/ROUND02_RESULTS.md)；逐项记录见 [实验日志](docs/EXPERIMENT_LOG.md)。

## 当前实现

- 原始 Tetouan 城市电力数据的读取、时间完整性检查和按时间 70%/15%/15% 划分。
- 只用训练集拟合标准化。每次预测只处理历史窗口，训练/验证目标不跨越各自边界。
- 当前原始输入路径跳过污染函数；历史污染实验用全局共享污染，未来评价目标保留原观测。
- SSA 单成分重构＋残差组；全部权重为 1 时能恢复输入。
- 小型预测代理F预训练；支持旧冻结模式，以及共享特征、头参数隔离的匹配冻结／交替协同训练。
- 首要评价使用各分支配对训练的同结构MLP；独立Ridge用于单独报告跨模型迁移。
- 支持完全关闭修复、固定保留阶数和逐组成分消融；配置、数据审计、权重、预测、指标、耗时与图形保存。

**SRD 状态：新增 `java` 后端直接调用固定作者提交的JAR，方法标为 `srd_java`，已与作者自带smoke逐点核验一致。** 旧Python实现仍标为 `srd_prototype`，实测与作者输出不同，不能称为原版SRD。原文与公开代码存在修复机制差别，本工程明确采用所记录的作者公开代码版本。见 `PROVENANCE.md`。

## 运行

Windows PowerShell，进入工程后：

```powershell
./scripts/run.ps1 -Mode tests
./scripts/run.ps1 -Mode ssa_static
./scripts/run.ps1 -Mode ssa_only
./scripts/run.ps1 -Mode ssa_structure
./scripts/run.ps1 -Mode task_alignment
./scripts/run.ps1 -Mode scale_fair
./scripts/run.ps1 -Mode mlp_primary
```

`ssa_static` 只运行原始输入的固定 SSA 诊断，不需要 TensorFlow/Java；`ssa_only` 先完成两种任务训练各三个种子，再运行固定诊断。均不评价测试集。详见第五轮记录中的复现与汇总命令。

`ssa_structure` 只分析原有36个训练历史的波形、频谱与加权相关性，不构造未来目标或验证/测试历史，不运行预测器、神经训练或修复。

`task_alignment` 按锁定协议使用全部可用日步长训练/验证窗口，比较成分聚合与完整分解、原头与TASK式头，并运行固定处理、缩放和代理预算对照。无人工污染、无SRD、无测试集评价。新旧窗口集合不同，分数不可直接跨轮混比。

`scale_fair` 复用已存在的 `runs/task_alignment_initial`，只重新拟合预测读出。标准化每折只拟合内训练前缀，Ridge选参不使用外层验证分数；上游模型冻结且未按内折重训。来源不存在会报错，不自动重训神经网络。

`mlp_primary` 按新的200拟合/51选检查点/54评价角色训练相同MLP主比较，提供同预算原始输入、固定缩放、固定SSA及学习全局尺度对照。所有分支共同初始化、相同代理更新预算；模型选定后再评价，不运行Ridge、不构造测试窗口。

以下入口保留历史配置，其中包含合成输入或人工污染，不作为本轮设置：

```powershell
./scripts/run.ps1 -Mode smoke
./scripts/run.ps1 -Mode pilot
./scripts/run.ps1 -Mode round02
```

原版Java入口（首次先配置并验证，再运行）：

```powershell
./scripts/run.ps1 -Mode java_setup
./scripts/run.ps1 -Mode java_verify
./scripts/run.ps1 -Mode java_pilot
```

`java_pilot` 默认共享任务特征、k=6，仅评价验证集；`java_round` 运行两种输入条件和k=3/6/9的六组对照。配置和依赖仅写入项目内，不改全局Java环境。源码/JAR校验失败会报错，不自动回退到原型。

协同机制的配对实验：`./scripts/run.ps1 -Mode collaborative`。两组均固定15轮选择头训练、AdamW/AMSGrad及梯度裁剪；协同组每批追加代理更新，包含共享特征。按重构验证代理MSE选同一轮的代理和头，独立Ridge不参与选检查点。原有 `legacy_frozen` 结果保留，不与新的匹配冻结组混为一谈。

`smoke` 明确使用合成周期序列；`pilot` 使用真实 Tetouan CSV，数据缺失会报错，不会自动改用合成数据。初版最多取 36/12/12 个训练/验证/测试窗口，均匀分布在各自时间段，仅用于检查工程。

下载公开输入（需要联网）：

```powershell
python scripts/fetch_resources.py
```

自选 Python 路径：

```powershell
./scripts/run.ps1 -Mode pilot -Python '你的Python路径' -Out 'runs/my_first_pilot'
```

每次输出目录必须是新目录，不覆盖已有结果。运行脚本优先使用本工程 `.venv`；没有时可只读复用本机已有 EEG 环境，不安装或修改其中依赖。工程及生成数据均写在本目录。

独立环境建议 Python 3.10/3.11，安装本项目及 `learning` 可选依赖。TensorFlow 是学习阶段依赖；`--fixed-only` 可跳过它，仅比较四组固定处理。

## 历史六组含义

| 运行名称 | 处理 |
|---|---|
| raw | 污染历史直接预测 |
| ssa | 固定前 K 个 SSA 成分重构 |
| srd_prototype | Python 季节修复原型 |
| ssa_srd_prototype | 固定 SSA 后接修复原型 |
| task_ssa | 任务训练的 SSA 加权插件 |
| task_ssa_srd_prototype | 任务插件后接修复原型 |

主要看完整组合对单独修复、普通串联与单独任务插件的增量，不能只与 raw 比较。当前只是一个种子、一个独立下游、一个污染条件，未做完整调参和多模型确认。修复误差采用原观测作相对参考，不等价于真实绝对干净信号。

最初代理 F 在未额外注入污染的训练历史上预训练；选择器用污染历史与未来目标训练。这一参考条件需要在报告中保留，不能把代理预训练描述为完全没有相对参考信息。

## 输出与版本

- `data_audit.json`：数据来源、时间范围、分界、标准化参数及所用窗口位置。
- `config.json`：实际参数；`run_metadata.json`：实现限制、环境、Git 提交与耗时。
- `metrics.csv`：验证/测试预测误差及历史修复误差；数值较小不自动证明生理/业务意义。
- `component_weights.npz`、模型权重、预测文件及输入、处理、训练曲线图。

`data/` 与 `runs/` 不进入 Git，下载内容附 SHA256 清单。代码与小型报告已同步到用户提供的 [GitHub 仓库](https://github.com/akawindow/ssa)，不提交原始大数据、权重或凭据。

原计划中的无污染条件与多训练种子评价已在第二轮补充；第三轮通过直接调用作者Java并逐点核验接口，建立了可用原版基线。旧Python原型仍未改写成等价移植版本。

第二轮补充：`round02` 按 `configs/round02_protocol.json` 运行，只构造训练/验证窗口；污染种子固定为 42，训练种子为 42/43/44。原卷积选择器有 161 个可训练参数；两个特征选择头均有 33 个可训练参数，分别使用预训练/随机冻结特征。共享特征选择器还包含冻结的特征提取权重，不能把 33 当成整个插件的总参数或计算成本。

当前判断：联合MLP的受控主比较和独立Ridge迁移诊断均已完成，二者目前都未显示任务选择器的稳定增量。后续先检查合格任务主干及成分选择的实际贡献，不因换评价器而把负结果改写为正结果。主次评价原则保存在研究决策和AGENTS.md，各轮记录按当时配置范围保留。

结果汇总可用同一个 Python 环境运行（输出目录必须为新目录）：

```powershell
python scripts/summarize_round02.py --run runs/round02_initial --out docs/results/round02_recheck
python scripts/diagnose_initial_heads.py --run runs/round02_initial --out docs/results/round02_initial_heads_recheck
```

第二条是看到权重记录后补做的诊断：代理已经预训练，分别评价权重头训练前后；不进行新的神经网络训练，不评价测试集。

## GitHub 版本管理

远程仓库为 [akawindow/ssa](https://github.com/akawindow/ssa)，分支为 `main`。代码与小型报告进入版本管理，数据、缓存、模型权重及运行目录保留本地。
