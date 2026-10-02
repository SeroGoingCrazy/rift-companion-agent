# 量化与 CPU 性能基准（DEV_SPEC I10）

2026-10-02。模型：I9 选定的 SFT r003 0.6B + 规则扰动 DPO β=0.3（`rift-slot-0.6b-dpo_rule_b03-f16.gguf`）。
机器：AMD Ryzen 5 9600X（6 核 12 线程），WSL2，CPU 编译的 llama.cpp `5fc4f3c`，`llama-server -t 6 -c 4096 -np 1`。
四个变体在**同一次运行**中依次加载、预热、测速、跑 L2（`training/scripts/quantize/bench.py`），原始结果见
`bench.md` / `bench.json`，文件 sha256 见 `manifest.json`，逐条 L2 报告为 `eval/reports/quant_<variant>_06b_slot_*`。

## 结果

| 变体 | 大小 | L2 主集 | L2 holdout | 合计 | 相对 F16 | E2E P50 / P95 | TTFT P50 | decode | 峰值 RSS |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| F16 | 1143 MB | 93.3%（56/60） | 88.9%（24/27） | 80/87 | 基准 | 901 / 1872 ms | 160 ms | 34 tok/s | 1686 MB |
| **Q8_0** | **610 MB** | **93.3%（56/60）** | **88.9%（24/27）** | **80/87** | **逐字相同** | **687 / 1167 ms** | 190 ms | **57 tok/s** | 1152 MB |
| Q4_K_M | 378 MB | 86.7%（52/60） | 88.9%（24/27） | 76/87 | −5 / +1 | 465 / 798 ms | 134 ms | 79 tok/s | 1128 MB |
| Q4_K_M + imatrix | 378 MB | 88.3%（53/60） | 88.9%（24/27） | 77/87 | −6 / +3 | 486 / 824 ms | 146 ms | 79 tok/s | 1128 MB |

- F16 复现了 I9 的 93.3% / 88.9%，87 条输出与 I9 报告逐字相同，说明本次评测与前几轮可比。
- 速度阶段的流式输出与质量阶段经线上抽取器得到的输出 87 × 4 条逐字一致（temperature 0 可复现）。
- Q4_K_M 两次独立运行（量化后第一次跑 3 个变体、加入 imatrix 后再跑 4 个变体）结果完全相同（52/60、24/27）。

## 验收：Q4_K_M 未通过

门槛是 Q4_K_M 比 F16 多错 ≤ 2 条。普通 Q4_K_M 多错 **4** 条，imatrix Q4_K_M 多错 **3** 条，**都未通过**。
imatrix 是按规格预先列出的补救手段，校准文本只取 SFT v0.4 的 train 部分（256 条，种子 42，按训练模板渲染，
`llama-imatrix` 校准 PPL 2.02），L2 评测集没有参与。把门槛的候选从 `q4_k_m` 改成 `q4_k_m_imat` 是在看到普通
Q4_K_M 失败之后做的，这里如实记录；改了之后同样未通过。之后没有再试 Q5_K_M / Q6_K：87 条评测上的逐个尝试，
最后挑出"恰好通过"的那个，等于用评测集选模型。

**决定（用户确认）：发布 Q8_0 作为线上模型。** 它在质量上与 F16 无损，体积减半，E2E P50 比 F16 快 24%。
`config/settings.yaml` 的 `local_slot.model` 改为 `rift-slot-0.6b-q8_0`。Q4_K_M 保留为备选，不发布。

### Q4_K_M 错在哪

| 样本 | F16 | Q4_K_M | Q4_K_M + imatrix | 说明 |
|---|---|---|---|---|
| main-005 "约个打野，单双排上分…" | ✓ | ✗ 漏 `service_type` | ✗ 同左 | DPO 刚修好的"长句漏键"又出现 |
| main-060 "找个会玩辅助位的小哥哥陪我单排上分…" | ✓ | ✗ 漏 `service_type` | ✗ 同左 | 同上 |
| main-033 "最后那个看着不错" | ✓ | ✗ 照抄原文 | ✗ 同左 | 候选引用照抄（DPO 的 `name_literal` 目标） |
| main-036 "1号吧" | ✓ | ✓ | ✗ 照抄"1号吧" | I9 中 DPO 修好的样本，imatrix 版退回 SFT 的错误 |
| main-004 "有没有温柔一点会聊天的陪玩…" | ✓ | ✗ 凭空多出 `companion_gender: female` | ✓ | |
| main-048 "确认" | ✓ | ✓ | ✗ `turn_intent: confirmation`（枚举越界） | |
| holdout-023 "不了，换一个人吧" | ✓ | ✗ 拒绝时多出名字 | ✓ | 拒绝换人时 delta 应为空（H5 起的老问题） |
| holdout-020 "帮我查一下明天去上海的航班" | ✓ | ✓ | ✗ unrelated → consult | |
| main-008 "男生女生都行" | ✗ `"male or female"` | ✗ | **✓ `any`** | I9 中 DPO 引入的退化，imatrix 版反而回到 SFT 的正确输出 |
| main-012 列表替换 | ✗ | ✗ | **✓** | |
| holdout-027 "打中路的" | ✗ mid→adc | **✓** | **✓** | I9 中 DPO 引入的退化 |

规律很清楚：**4 bit 量化抹掉的正是 DPO 的那几步微调**。I9 中规则 DPO 只移动了 5 个样本（+3 / −2），β 从 0.1 到 0.3
都不改变 argmax，说明这些边界上的 logit 差距很小；Q4 的量化噪声足以把它们翻回去，修好的和弄坏的一起回到
SFT 的行为（main-036、main-008、holdout-027）。Q8_0 的噪声小一个量级，87 条输出与 F16 逐字相同。剩下的翻转
（漏 `service_type`、拒绝时多出名字、unrelated 判成 consult）是 SFT 本来就不稳的地方。要让 Q4_K_M 过门槛，应该在训练侧加大这些边界的
间隔（对比集 / DPO 增加漏键与照抄类样本），而不是继续换量化配置。

## 前缀预热

system prompt 在所有请求间共享（约 730 token），llama-server 的 `cache_prompt` 会复用它的 KV 缓存。
预热之后每个请求平均只需计算 79 个 prompt token（复用 731 个）：

| 变体 | 冷启动首请求 TTFT | 预热后 TTFT P50 | `cache_prompt: false` TTFT P50（10 条） |
|---|---:|---:|---:|
| F16 | 934 ms | 160 ms | 934 ms |
| Q8_0 | 1430 ms | 190 ms | 1362 ms |
| Q4_K_M | 730 ms | 134 ms | 745 ms |

预热为每个请求省下 0.6–1.2 s 的 TTFT，相当于 E2E 的一半以上，所以 J2 的启动预热有必要。Q8_0 的 prefill 比
F16 还慢（1362 vs 934 ms）：这个 CPU 版本的 F16 矩阵乘走了更快的路径，而 Q8_0 赢在 decode（访存减半）。
预热后 prefill 只剩约 80 token，decode 占了大头，所以 Q8_0 的 E2E 仍然比 F16 快。

## 其他发现

- **Windows 上的 `localhost`**：bench 最初用 `http://localhost:8080`，预热后 TTFT 仍有约 2.1 s。原因是 Windows 先尝试 `::1`，
  WSL 的端口转发只应答 IPv4，Python 的同步 httpx 每个请求都要等约 2 s 超时后才回落到 127.0.0.1。bench 改用
  `127.0.0.1` 后 TTFT 降到约 0.1 s。I6–I9 的 L2 延迟走的是线上抽取器（异步客户端），没有受影响：I9 rule β=0.3
  主集 P50 872 ms，本次 F16 为 897 ms。
- **线程数**：`-t 4 / 6 / 12` 小样本对比（8 条），decode 分别为 73 / 78 / 68 tok/s（Q4_K_M），取物理核数 6。
- 加载时间 F16 6.8 s、Q8_0 4.1 s、Q4_K_M 3.1 s（WSL 读 Windows 盘，冷读为主）。

## 复现

```bash
uv run python training/scripts/quantize/quantize.py
uv run --env-file .env python training/scripts/quantize/bench.py
```

配置：`training/configs/quantization/rift_slot_0.6b.yaml`。GGUF 位于 `training/outputs/gguf/`（git 忽略），sha256 见 `manifest.json`。
