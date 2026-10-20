# SSA 任务驱动去噪与季节清洗：初步实验工程

本工程独立于原 EEG 复现目录。目标是验证：任务训练的 SSA 选择器，在接入后续季节清洗和独立预测器后是否仍有价值。当前版本是用于排错的小规模探索，不是论文结论。

**推荐新实验使用作者 Java SRD 后端。** 本机已复用 JDK21/Maven3.9.9，作者代码构建及逐点接口核验通过，现已接回SSA与独立预测框架。见 [原版Java基线记录](docs/JAVA_SRD_RESULTS.md)。旧 `pilot` 和 `round02` 保留 Python 原型设置用于历史复查；新入口是 `java_pilot`、`java_round`。

已完成的首次运行、数值、适用范围和下一步见 [初步实验记录](docs/INITIAL_RESULTS.md)。

第二轮已补齐无额外污染对照、TASK 式共享特征与匹配随机特征对照，完成 18 次验证集运行及训练前后诊断。结果与下一步见 [第二轮实验记录](docs/ROUND02_RESULTS.md)；逐项记录见 [实验日志](docs/EXPERIMENT_LOG.md)。

## 当前实现

- 原始 Tetouan 城市电力数据的读取、时间完整性检查和按时间 70%/15%/15% 划分。
- 只用训练集拟合标准化。每次预测只处理历史窗口，训练/验证目标不跨越各自边界。
- 一条全局污染实现供所有组、所有重叠历史窗口共享；未来评价目标保留原观测。
- SSA 单成分重构＋残差组；全部权重为 1 时能恢复输入。
- 小型预测代理 F 预训练并固定；共享成分选择器接受预测误差训练。
- 下游用独立 Ridge 预测器，各方法按相同设置分别适配。
- 六组对照、数据审计、配置、环境、污染掩码、权重、预测、指标、耗时与图形保存。

**SRD 状态：新增 `java` 后端直接调用固定作者提交的JAR，方法标为 `srd_java`，已与作者自带smoke逐点核验一致。** 旧Python实现仍标为 `srd_prototype`，实测与作者输出不同，不能称为原版SRD。原文与公开代码存在修复机制差别，本工程明确采用所记录的作者公开代码版本。见 `PROVENANCE.md`。

## 运行

Windows PowerShell，进入工程后：

```powershell
./scripts/run.ps1 -Mode tests
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

## 六组含义

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

`data/` 与 `runs/` 不进入 Git，下载内容附 SHA256 清单。先用本地 Git 即可；目前不需要 GitHub。后续提供空仓库地址、完成官方登录后再配置远程，禁止把密码、访问令牌或原始大数据放进提交。

原计划中的无污染条件与多训练种子评价已在第二轮补充；第三轮通过直接调用作者Java并逐点核验接口，建立了可用原版基线。旧Python原型仍未改写成等价移植版本。

第二轮补充：`round02` 按 `configs/round02_protocol.json` 运行，只构造训练/验证窗口；污染种子固定为 42，训练种子为 42/43/44。原卷积选择器有 161 个可训练参数；两个特征选择头均有 33 个可训练参数，分别使用预训练/随机冻结特征。共享特征选择器还包含冻结的特征提取权重，不能把 33 当成整个插件的总参数或计算成本。

当前判断：作者Java基线已经跑通并接回框架，阈值对正常输入的改动与预测成绩有明显影响；共享特征组合仍只有很小的单种子增量，尚未证明稳定优势。下一步优先重复种子并保留正常输入对照，初始化及交替训练暂缓。详见最新Java记录，前两轮结论仍按当时原型范围保留。

结果汇总可用同一个 Python 环境运行（输出目录必须为新目录）：

```powershell
python scripts/summarize_round02.py --run runs/round02_initial --out docs/results/round02_recheck
python scripts/diagnose_initial_heads.py --run runs/round02_initial --out docs/results/round02_initial_heads_recheck
```

第二条是看到权重记录后补做的诊断：代理已经预训练，分别评价权重头训练前后；不进行新的神经网络训练，不评价测试集。

## 后续接入 GitHub

本地工程已有独立 Git 仓库。若要云端备份：在 GitHub 新建一个空的私有仓库，不勾选初始化 README、.gitignore 或 License；把该仓库 HTTPS 地址发来，并说明希望同步。若本机尚未登录，使用 GitHub 官方的浏览器授权流程或 Git Credential Manager 登录，密码及访问令牌不用发送到聊天。

当前没有配置远程仓库，也没有上传代码或数据。代码与小型报告进入版本管理，数据、缓存、模型权重及运行目录保留本地。
