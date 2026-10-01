# 纯大模型基线（DeepSeek，阶段 H5）

- 时间：2026-10-01；模型：`deepseek-chat`（JSON mode，temperature 0）
- 抽取器配置：`slot_extractor.primary = llm`，无降级、无影子（M1）
- 明细报告：`baseline_llm_slot_main.md` / `baseline_llm_slot_holdout.md`（L2）、`baseline_llm_e2e.md`（L4），
  同名 `.json` 含逐条样本 / 逐轮记录

## 指标（填入 DEV_SPEC 2.4「纯大模型」列）

| 指标 | 值 | 说明 |
|---|---|---|
| L2 主集通过率 | **98.3%**（59/60） | 协议通过率 100%，平均任务分 0.996 |
| L2 holdout 通过率 | **88.9%**（24/27） | 协议通过率 100%，平均任务分 0.966 |
| L2 单次抽取延迟 P50 / P95 | 775 / 1099 ms（主集） | 4 并发，含网络 |
| L4 任务完成率 | **100%**（22/22） | 两次独立运行结果一致 |
| L4 平均轮数 | 3.32 | 73 轮 / 22 个剧本 |
| L4 单轮延迟 P50 / P95 | 834 / 1686 ms | max 9.5 s 为首次咨询时 knowledge-mcp 冷启动（加载 reranker） |
| 远程 LLM 调用 / 会话 | 4.14（1.25 / 轮） | 只计 `llm:*` generation span |

L4 每轮远程调用按回复类型：IDLE 首轮 = 分类 + 抽取（2 次），预约中每轮 = 抽取（1 次），插话咨询 =
抽取 + 作答（2 次），订单管理确认 / 执行取消走规则（0–1 次）。小模型接管抽取后（M3），预约中的轮次
不再需要远程调用，这就是阶段 J 要对比的主要收益。

## DeepSeek 的错误类型（造数重点）

协议层零错误：JSON mode 加上严格 schema，没有出现多余字段、枚举越界或类型错误。4 个任务层错误
全部集中在「抽什么、不抽什么」：

| # | 样本 | 用户输入 | 期望 | DeepSeek 输出 | 类型 |
|---|---|---|---|---|---|
| 1 | main-023 | 风格没要求，段位改成大师 | `style_preference: "any"` | `style_preference: null` | any / null 混淆 |
| 2 | holdout-008 | 不指定阿狸酱了，换谁都行 | `companion_name: "any"` | `companion_name: null` | any / null 混淆 |
| 3 | holdout-023 | 不了，换一个人吧（待确认时） | `delta: {}`, `confirmation: no` | 另加 `companion_name: null` | 确认轮多抽键 |
| 4 | holdout-002 | 想找个声音好听的妹子… | `style_preference: "声音好听"` | 改抽 `voice_required: true`，漏掉风格 | 语义映射错误 |

据此给阶段 I 的场景规格（I2）定的重点：

1. **三态对比样本**：非必填字段（`style_preference`、`companion_name`、`companion_gender`、
   `voice_required`、`budget_per_hour`）的「不限 / 都行 / 没要求 / 换谁都行」→ `"any"`，与
   「先不定 / 再想想 / 先别管」→ `null` 成对出现。DeepSeek 在必填字段上分得清，在非必填字段上倾向输出
   `null`（4 个错误里占 2 个）。
2. **确认 / 拒绝轮次只输出 confirmation**：「换一个人」「算了」只表示 `no`，delta 为空；换人由代码
   回到候选列表，模型不应额外撤回 `companion_name`。
3. **声音相关说法区分**：「声音好听 / 嗓音甜」是风格偏好，「要开麦 / 能语音」才是 `voice_required`；
   两者同时出现的混合样本也要覆盖。
4. 首轮多槽位、相对时间、黑话、候选引用在 DeepSeek 上全部正确，小模型仍需按 3.7 的配额覆盖，但不必额外加权。

## 局限

- L2 每类只有 3–8 条，holdout 的 3 个错误就让通过率差了 10 个百分点；比较配置时应同时看逐字段错误分布，
  而不只看通过率。
- L4 剧本的话术接近标准说法，DeepSeek 全部通过，说明当前剧本对大模型的区分度不高；小模型接入（J5）后
  它主要用来回归降级链路。后续可以加入口语化、错别字和多意图混合的剧本，提高难度。
- DeepSeek 在 temperature 0 下也不是严格确定性的；L2 只跑了一次，没有多次运行的方差统计。
