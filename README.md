# 语音活动区间分析器（VAD Analyzer）

本地真实运行的语音活动检测 / 区间分析工具，面向“录对白 → 交给剪辑器”的场景。
它从录音中判断**哪里存在语音活动**，输出采样级精确的时间范围、置信信息，
并能导出可审听、经逐样本核验的 WAV 片段。

- 模型与规则全部在**本机用 numpy 真实计算**，没有伪造的执行链、没有联网推理。
- **帧级判定**（`detector.py` / `features.py` / `model.py`）与**区间合并**
  （`segments.py` / `stream.py`）严格分离。
- 明确区分“有声音 / 语音样活动判定”与“确认是人在说话”：本工具输出的是
  **语音活动（voice activity）区间**，不是说话内容识别，也不是说话人分离。
  风噪、音乐、键盘声可能在声学上类似语音，工具会尽力压制但无法保证通用正确
  （见文末“已知局限”）。

---

## 1. 图形入口（主要入口）

文档首要列出的就是**语音区间图形窗口**。在带桌面环境（或 X11 / Xvfb）的机器上：

```bash
cd /workspace
python3 -m vad_analyzer gui
```

如果系统 Python 没有 `tkinter`（例如精简 Debian），本仓库自带了一份本地 Tk 兜底，
直接用封装脚本即可：

```bash
bash run_gui.sh
```

正常 Debian / Ubuntu 系统推荐直接装系统包后用第一条命令：

```bash
sudo apt install python3-tk pulseaudio-utils   # 播放后端可选
pip install -r requirements.txt
```

图形窗口（“声音分析 / 语音区间”窗口）提供：

- 多轨画布：波形、帧概率曲线（含进入/退出阈值线）、**原始帧判定**、
  **最终区间**、**漏检片段**、参考标注、锁定切点。
- 6 个可调参数滑杆（进入阈值、退出阈值、最短语音、间隙容忍、边界扩展、
  强帧密度），改动后用缓存的帧判定**立即重新合并**，不必重算特征。
- 播放整段 / 播放选中区间 / 停止（无音频后端时在界面红条明确报错，不静默）。
- 导出选中区间、导出全部区间，每个片段导出后**重新读回 WAV 逐样本核验**。
- 五个结果页签：**活动区间、漏检片段、锁定切点冲突、导出核验、指标日志**。
  活动区间与漏检片段都来自当前音频的**实际运算**，不是占位文本。
- “保存标注草稿…”：把当前区间写成待人工复核的标注 JSON
  （`annotator` 字段固定为 `vad-draft-needs-human-review`，**不会**冒充人工标注）。
- “载入锁定切点…”：载入剪辑器已锁定的切点，只显示冲突，**绝不移动锁点**。
- 打开任意 WAV，或一键载入内置示例对白 `data/labeled_short/interview_dialogue.wav`。

> 容器 / 无头环境可用 `Xvfb :99` 后 `DISPLAY=:99 python3 -m vad_analyzer gui`。
> 自动化 GUI 冒烟测试会真实创建 Tk 窗口并加载示例、改阈值、切页签。

---

## 2. 命令行（仅辅助入口）

CLI 用于批处理与自检，**不是主要入口**；日常审听与调参看上面的图形窗口。

```bash
# 一键自检：分岔撤回、锁定切点冲突、导出逐样本核验
python3 -m vad_analyzer demo

# 分析一个 WAV：区间 + 漏检 + 锁点冲突 + 导出并核验 + JSON
python3 -m vad_analyzer analyze data/labeled_short/interview_dialogue.wav \
    --cuts data/labeled_short/interview_locked_cuts.json \
    --annotation data/labeled_short/interview_dialogue.json \
    --export-clips exports/cli \
    --json-out reports/cli_analysis.json

# 可调参数（均可选，默认取本地模型调优配置）
python3 -m vad_analyzer analyze speech.wav \
    --threshold 0.375 --exit-threshold 0.215 \
    --min-speech-ms 130 --min-strong-ratio 0.35 \
    --gap-ms 600 --pad-ms 90

# 重建合成数据集 / 重训本地模型 / 跑独立评估
python3 -m vad_analyzer generate-data --root data --train-n 8 --eval-n 4 --val-n 6
python3 -m vad_analyzer train    --train-dir data/train
python3 -m vad_analyzer evaluate --eval-dir data/eval --out reports
```

`analyze` 会自动寻找同名 sidecar 标注（`x.json`、`x.annotation.json` 或
`annotations/x.json`），用 `--no-sidecar` 关闭。运行
`python3 -m vad_analyzer --help` 或 `... analyze --help` 查看全部参数。

---

## 3. 快速验证（测试）

```bash
bash run_tests.sh          # 纯逻辑 / 音频 / 指标 / 场景，28 项
DISPLAY=:99 bash run_tests.sh --gui   # 额外执行真实 Tk GUI 冒烟
```

当前共 **29 个测试全部通过**（不带 `--gui` 时 GUI 用例 skip）。
测试覆盖：区间合并语义、实时撤回、导出全样本核验、标注差集与指标、
锁定切点不可变、分岔 A/B 场景、三类干扰零误区间、GUI 真实加载重算。

---

## 4. 端到端真实执行链与目录

```
WAV(int16/mono)
  └─ audio.load_wav                标准库 wave，规范成 mono int16，分析转 float32
      └─ features                  25ms 窗 / 10ms hop，13 维声学特征
          └─ model.LocalModel      本地 numpy 逻辑回归（model_weights.json）
              或 HeuristicScorer   权重缺失时的规则兜底
          └─ detector.FrameDetector   逐帧概率 + 原始二值判定（帧级，独立）
              ├─ segments.merge_probabilities  离线区间合并（区间级，独立）
              └─ stream.StreamingVad           实时事件：暂报/撤回/确认
                  └─ pipeline.analyze_audio    区间 + 漏检差集 + 锁点冲突
                      ├─ exporter.export_interval  直接切 int16，重读逐样本核验
                      ├─ evaluate                  事件/覆盖/边界/分桶指标
                      └─ gui / cli                 两个真实入口
```

目录：

| 路径 | 内容 |
| --- | --- |
| `vad_analyzer/` | 全部源码（特征、模型、检测、合并、流式、导出、锁点、评估、CLI、GUI） |
| `vad_analyzer/model_weights.json` | 本地训练的逻辑回归权重与标准化参数；后端串 `local-logistic-v1:model_weights.json` |
| `data/labeled_short/` | 11 个固定人工复核向短片段（含分岔 A/B、对白、重叠、三类干扰、低 SNR）及同名 JSON |
| `data/train/` `data/validation/` `data/eval/` | 合成并精确标注的训练 / 验证 / **独立评估集**（WAV + JSON） |
| `reports/metrics_summary.json` `metrics_report.md` `event_metrics.csv` | 可复核指标 |
| `reports/failure_cases/` | 16 个保留的失败 / 低 SNR / 音乐床案例（每案例一个目录） |
| `exports/demo` `exports/cli_test` `exports/gui_smoke` | 导出片段及核验结果 |
| `tests/` | unittest 测试 |
| `run_gui.sh` `run_tests.sh` | 注入本地 numpy / Tk 兜底路径的运行脚本 |
| `.extradeps/` `.pydeps/` | 无 root 环境下的本地 Tk / Python 依赖兜底（可删后用系统包替代） |

---

## 5. 帧级判定 与 区间合并 的分离

这是刻意的架构边界：

1. **帧级**（`features.py` + `model.py` + `detector.py`）只对每个 10ms hop
   输出一个 `[0,1]` 语音样概率和原始二值判定，**不知道“段”的存在**。
   13 维特征包括：对数 RMS、相对本底增益、过零率、低/中/高频带比、
   谱质心、谱平坦度、滚降、谱通量、浊音(voicing)、2–8Hz 音节调制比、包络变异系数。
   因此**不是“声音大就是语音”**：幅度只是 13 个特征之一，最终由多维模型 /
   规则共同决定。

2. **区间级**（`segments.py` 的 `EvidenceFSM` / `cores_to_intervals`）只消费
   概率序列与帧起始样本，负责迟滞、间隙桥接、最短证据门控、边界扩展。
   同一份帧判定可以在 GUI 里拖动滑杆反复重新合并，无需重算特征。

---

## 6. 参数语义（避免把句中短停顿全切碎）

运行默认值来自 `pipeline.config_for(detector)`（模型阈值选定为 0.375）：

| 参数 | 默认 | 准确语义 |
| --- | --- | --- |
| `enter_threshold` | **0.375** | 概率 ≥ 此值算一帧**进入/强证据**，可开启区间或重置负证据计数 |
| `exit_threshold` | **0.215** | 活动中概率低于此值才算明确退出证据；介于两者之间是弱/尾音证据 |
| `gap_tolerance_ms` | **600** | 连续多少毫秒低于进入阈值才真正闭段；**短于它的换气/停顿被桥接成同一段**，更长停顿才分段 |
| `provisional_ms` | **220** | 负证据持续到此时长可先“暂报可能结束”（必须 < gap） |
| `exit_tail_ms` | **140** | 只有距末个强帧在此范围内的弱证据才延长存活尾，防止长段中等概率背景把终点拖过整个间隙 |
| `min_speech_ms` | **130** | 候选必须满足“**首强帧 → 末强帧**”的**强证据跨度**至少这么长；不是按扩展后的终点算，避免一声点击借尾音蒙混 |
| `min_strong_ratio` | **0.35** | 核心内 ≥ enter 阈值的强帧占比下限；稀疏键盘簇即使偶有高概率帧也会因密度不足被丢弃 |
| `edge_pad_ms` | **90** | 交付区间在检测核心两端**额外保留**的余量（采样级 clamp 到文件范围），用于保留换气与被音乐盖住的尾音；padding 不移动检测核心 |
| `min_confidence` | 0.0 | 可选的段平均置信下限，默认关闭 |
| 帧长 / hop | 25ms / 10ms | 分析网格 |

“最短语音”和“强帧密度”是两道独立门控：前者卡强证据的**时间跨度**，
后者卡强证据的**密集程度**，专门压制键盘这类稀疏瞬态簇。

### 两人换气短暂重叠 / 尾音与背景的权衡

重叠本身不需要特殊开关：两个话轮在声学上连成一片时，FSM 依据
**进入证据**（新的强帧）与**退出证据**（持续低概率）决定是桥接还是分段。
`gap_tolerance_ms` 负责“换气桥接”，`edge_pad_ms` + `exit_tail_ms` 负责
“尾音保留”，而强证据门控防止为了收尾音而把背景声整段并入。
评估时对重叠参考话轮先用 `labels.union_spans` 合并成**声学活动区**
（VAD 不做说话人分离），同时仍单独统计短语级 split/merge。

---

## 7. 实时语义：暂报、撤回、归档

`stream.StreamingVad` 在离线同一套 `EvidenceFSM` 上产生事件：

```
speech_started
  → provisional_end     （负证据达到 provisional_ms：可先“暂报可能结束”）
      → end_retracted   （暂报后、真正闭段前又来强证据：撤回暂报，合并为同一段）
      或 segment_confirmed（负证据达到 gap_tolerance_ms：证据齐全，归档切片）
  → short_dropped       （闭段后未通过最短/密度门控，丢弃并说明原因）
```

关键不变量（有专门测试）：

- **撤回只可能发生在“确实开着且已暂报”的段上**；一旦已经 `close`，
  之后的强帧属于新段 `speech_started`，不会对一个已归档段发假撤回。
- 归档切片只在 `segment_confirmed`（证据齐全）后产生，导出长度与
  `_processed_samples` 对齐，不会拿到尚未到达的样本。

### 专门验收：相同前缀、末尾半秒分岔

`scenarios.build_fork_pair()` 构造两段**前缀逐样本完全相同**、在约 2.6s
（最后约半秒）分岔的音频：

- **A：分岔后又出现语音** → 先 `provisional_end`，随后一次
  `end_retracted`，最终只有 **1 个** `segment_confirmed`，且终点延伸到音频末尾附近。
- **B：分岔后只有安静房间声** → 有 `provisional_end` 但**没有**
  `end_retracted`，正常确认 1 段，终点不超过分岔点 +120ms。

对应测试：`tests/test_scenarios.py::ForkScenarioTests`
与 `tests/test_stream.py::RetractionTests`；命令行
`python3 -m vad_analyzer demo` 会真实跑一遍并打印 `DEMO_OK`。

---

## 8. 已锁定采访切点：只报告冲突，不能移动

`cuts.py` 的 `LockedCut` 是冻结数据类（赋值会抛 `FrozenInstanceError`）。
`evaluate_locked_cuts(cuts, intervals)` 把每个锁点与检测区间比较，给出：

- `error / cut_through_voice_activity`：锁点落在语音**核心**内，切下去会切字；
- `warning / inside_edge_padding`：锁点落在边界扩展余量内，可能切掉保留的气口/尾音；
- `ok / clean_gap`：锁点位于干净间隙。

系统**没有任何代码路径会改写锁点 sample**；GUI 的“锁定切点冲突”页签和
CLI 输出都只呈现冲突。示例 `data/labeled_short/interview_locked_cuts.json`
含 4 个锁点（核心 error、pad warning、干净 gap、末段核心 error）。
`tests/test_cuts.py` 验证核心命中报 error 且锁点样本前后不变。

---

## 9. 导出片段与“首尾样本”全样本核验

`exporter.export_interval` 直接从原始 int16 数组按
`[start_sample, end_sample)` 切片写 WAV（不经过 float 重采样，保证采样精确）。
随后 `verify_export` **重新用 `wave` 读回导出文件**，逐项核验：

- 实际采样数 == `end_sample - start_sample`；
- 读回的**每一个样本**都与源区间对应样本一致（`all_samples_match`，非抽样）；
- 第一个 / 最后一个样本的期望值与实际值一致（含双方 sha256）。

返回的 `ClipVerification.verified` 仅在
`length_match and all_samples_match and edges_match` 全为真时成立。
GUI“导出核验”页签与 CLI 都会打印每个文件的首/末样本与 `verified=True/False`。
`tests/test_exporter.py` 用随机 48kHz 数据验证全样本、长度、边缘与独立复核。

---

## 10. 数据、标注与人工复核工作流

标注 JSON schema（与生成数据一致）：

```json
{
  "audio_file": "speech.wav",
  "sample_rate": 16000,
  "segments": [
    {"start": 1.20, "end": 3.40, "kind": "speech", "speaker": "A", "text": "…"}
  ]
}
```

- 时间可用秒（`start`/`end`），也可用采样（`start_sample`/`end_sample`）。
- 生成器另写 `active_spans_samples` 作为帧级精确真值；没有它时评估回退到段边界。
- 真实 WAV + 人工 / 人工修正的 JSON 可直接用于 `analyze --annotation` 与评估。
- GUI“保存标注草稿…”产出同 schema 文件，但
  `annotator = "vad-draft-needs-human-review"`，并在 `note` 中明确需人工修正；
  请人工核对 start/end 后**另存为正式标注**，不要把草稿当人工真值汇报。

数据类别（每类 4 条独立评估片段，共 36 条 / 235.5s）：
`clean_speech`、`breath_gaps`、`overlap_speech`、`noisy_speech`、
`low_snr_speech`（含 SNR 2 / -2 dB）、`music_speech`、
以及纯干扰 `interf_wind` / `interf_keyboard` / `interf_music`。

---

## 11. 评估指标（不只报总体准确率）

完整结果见 `reports/metrics_report.md`、`metrics_summary.json`、
`event_metrics.csv`。当前独立评估集上的关键数字（本地模型后端）：

**先看口径警告**：评估集整体静音占比 **68.4%**，因此帧级 accuracy=0.879
会被长静音明显抬高，**不能作为主要结论**。应优先看漏检、误触发与事件指标。

- **语音时长漏检率（pooled voice miss）≈ 0.02%**：参考语音样本中未被任何
  检出覆盖的比例（按真实区间差集计算）。
- **误触发时长 6.35 秒/分钟**（非语音被误判为活动的时长速率）。
- 帧级：speech miss 2.77%，非语音 FPR 16.44%（FPR 偏高主要来自音乐床，见局限）。
- **严格事件指标**（参考声学活动区先 union 重叠话轮，贪心 1:1，IoU≥0.5）：
  参考 60 / 检出 47 / 匹配 35，recall 0.583、precision 0.745、F1 0.654。
  这里大量“漏事件”其实是**合理桥接 / 重叠合并**导致的粒度差异，而非丢声音，
  所以同时提供下面的覆盖口径。
- **多对一 coverage 口径**（参考 ≥50% 被覆盖、检出 ≥50% 有据）：
  recall **1.00**、precision **0.979**。它回答“声音有没有被保住 / 检出段有没有依据”，
  与严格 1:1 的粒度指标并列，不互相掩盖。
- **边界误差（核心边界，平均绝对误差）**：起点 ≈ **118ms**、终点 ≈ **70ms**；
  交付边界因含 90ms pad 会再大约一个 pad 量（这是有意保留的尾音余量）。
- **按事件时长分桶**（`<1s` / `1–2.5s` / `≥2.5s`）分别给事件 recall、
  起终点 MAE 与 p95，避免长片段平均掩盖短事件边界问题。短桶 recall 0.714、
  起点 MAE 36ms / 终点 63ms；中桶 recall 0.549（主要被音乐床与桥接口径拉低）。
- split / merge 事件数按短语图单独统计（VAD 不分离说话人）。
- **干扰专项**：curated 的 wind / music 纯干扰在默认配置下 **0 个误事件**，
  keyboard 评估集中 4 条有 1 条产生 1 个误事件；单元测试对固定 seed=5 的
  wind/keyboard/music 断言最终区间必须为空。

`reports/failure_cases/` 原样保留了 **16 个失败案例目录**，包括全部
`music_speech_*` 与多数 `low_snr_speech_*`，每个目录含 WAV、标注、
指标与（如有）对比信息，供复核而不是删除藏拙。

---

## 12. 验收点 → 代码 / 测试追溯

| 要求 | 实现位置 | 验证 |
| --- | --- | --- |
| 两段相同前缀、末尾分岔；A 撤回暂报合一段，B 无后续 | `scenarios.build_fork_pair`、`stream.StreamingVad` | `tests/test_scenarios.py`、`tests/test_stream.py`、CLI `demo` |
| 已锁采访切点不能移动，只报告冲突 | `cuts.py`（`LockedCut` 冻结、`evaluate_locked_cuts`） | `tests/test_cuts.py`、示例锁点 JSON、GUI 冲突页签 |
| 导出后重读 WAV 核验实际首/末样本（全样本一致） | `exporter.py.export_interval/verify_export` | `tests/test_exporter.py`、`demo`、GUI 导出核验页签 |
| 两人换气重叠，按进入/退出证据合段 | `segments.EvidenceFSM`、`labels.union_spans` | `tests/test_segments.py`、`test_labels_metrics.py`、`overlap_turns` 样例 |
| 风噪/音乐/键盘干扰，不靠“声音大” | `features.py` 13 维特征 + 本地模型、强证据/密度门控 | `InterferenceRejectionTests`、eval `interf_*` 类别 |
| 帧判定与区间合并分离；参数语义清晰 | `detector.py` vs `segments.py` | `tests/test_segments.py` 各参数精确断言 |
| 最短语音 / 间隙容忍 / 边界扩展 | `VadConfig`、`cores_to_intervals` | 短 blip 丢弃、短/长 gap、pad 采样精确测试 |
| 标注短录音 + 独立评估集；漏检/误触发/边界；低 SNR 失败案例 | `datasets.py`/`synth.py`、`evaluate.py`、`data/`、`reports/failure_cases/` | `test_labels_metrics.py` + 已生成报告 |
| 按语音时长分桶 + 按事件的可复核指标 | `evaluate._bucket_rollup`、`event_metrics.csv` | `metrics_report.md` |
| GUI 显示原始判定与最终区间、调阈值、播放、结果区、异常提示 | `gui.py` | `tests/test_gui_smoke.py`（Xvfb 真实控件） |
| CLI 仅辅助；文档列出图形入口 | `cli.py`、`__main__.py`、本 README 第 1 节 | `python3 -m vad_analyzer --help` |

---

## 13. 依赖与环境

- Python 3.11（3.10+ 应可运行），核心计算只依赖 `numpy>=1.24`。
- 读写 WAV 用标准库 `wave`，无需 scipy / soundfile。
- GUI 用标准库 `tkinter`；正常系统 `sudo apt install python3-tk`。
  无 root 精简环境可用仓库自带 `.extradeps/` + `run_gui.sh` / `run_tests.sh`。
- 播放为可选：优先 `sounddevice`，其次 `simpleaudio`，再次系统命令
  `paplay` / `aplay`（Debian: `pulseaudio-utils` / `alsa-utils`）。
  都不可用时 `AudioPlayer.available()` 为假，GUI 点播放会明确报错，
  但分析、导出、核验不受影响（导出的 WAV 可拿到任何机器审听）。

---

## 14. 已知局限（请如实转述，不要过度宣称）

1. **“语音活动”不等于“确认是人在说话 / 说了什么”**。输出标签是
   voice activity / voice-activity-like；节奏性音乐、某些噪声仍可能误判。
2. 当前随仓库提供的“语音”数据主要由 `synth.py` 的共振峰 / 谐波模型合成，
   标注是**精确机器真值、可人工复核**，尚不是大规模真人手工标注。
   管线支持真实录音与人工 JSON，但在真实口音 / 房间 / 远场条件下的泛化
   需要用真人标注集重新 `train` + `evaluate` 后才能宣称。
3. **音乐床 + 语音（`music_speech`）是当前主要失败类型**：节奏性音乐会让
   FSM 过度桥接，严格事件指标差（4 条共 9 漏 / 4 误），但 coverage≈1.0，
   即声音内容基本没丢、只是粒度被并成大段。全部案例保留在
   `reports/failure_cases/music_speech_*`。
4. 极低 SNR（如 -2dB）下帧漏检与边界误差明显增大，相关案例同样保留未删。
5. VAD **不做说话人分离、不识别文本**；重叠话轮按声学活动区合并评估。
6. 本开发容器没有音频输出设备，无法由作者肉耳验证播放；播放代码路径真实，
   且导出 WAV 已通过逐样本核验，可在有声卡的机器上直接审听。
