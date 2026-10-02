# models/

本地槽位抽取器的模型文件放在这里。目录被 git 忽略（只提交本说明），GGUF 不进仓库。

| 文件 | 说明 |
|---|---|
| `rift-slot-0.6b-q8_0.gguf` | 发布模型（Q8_0，639 446 816 字节） |
| `MODEL_CARD.md` | 模型卡（复制自 `training/MODEL_CARD.md`：训练数据版本、评测结果、许可证） |

发布模型的 sha256：

```
8b197e753a62fb9a4853c70f2928db6e12d9a69f490c537daf480eae229573e4  rift-slot-0.6b-q8_0.gguf
```

全部变体（F16 / Q8_0 / Q4_K_M）及其 sha256 见 `training/experiments/quant/manifest.json`。只发布 Q8_0：Q4_K_M
没有通过 I10 的质量门槛（`training/experiments/quant/report.md`）。

## 1. 获取模型

模型目前没有公开托管，三选一：

- **从训练机复制**：`training/outputs/gguf/rift-slot-0.6b-q8_0.gguf` → `models/`。
- **重新构建**：按 DEV_SPEC I7–I9 训练、合并得到 `rift-slot-0.6b-dpo_rule_b03-f16.gguf`，再运行
  `uv run python training/scripts/quantize/quantize.py --variants q8_0`，产物在 `training/outputs/gguf/`。
- **HuggingFace（可选）**：需由所有者确认许可证与条款（见模型卡"许可证与使用限制"）后手动上传；上传后用
  `huggingface-cli download <repo> rift-slot-0.6b-q8_0.gguf --local-dir models` 下载到本目录。

放置与校验（仓库根目录）：

```bash
cp training/outputs/gguf/rift-slot-0.6b-q8_0.gguf training/MODEL_CARD.md models/
```

```bash
sha256sum models/rift-slot-0.6b-q8_0.gguf
```

输出必须与上面的 sha256 一致。

## 2. 启动 llama-server

任意较新的 llama.cpp 都可以（I10 测试用的是 CPU 版 `5fc4f3c`），可以用官方 release，也可以自己编译。
**必须关闭 thinking**：训练用的是 `enable_thinking: false`，assistant 回合以空 think 块开头。

```bash
llama-server -m models/rift-slot-0.6b-q8_0.gguf -a rift-slot-0.6b-q8_0 --host 0.0.0.0 --port 8080 -c 4096 -np 1 -t 6 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

`-t` 设为物理核数。`curl http://127.0.0.1:8080/health` 返回 `{"status":"ok"}` 即表示模型已加载。

启动后先发一次请求预热，让共享的 system prompt（约 730 token）进入 KV 缓存；之后每个请求只需计算自己的
用户部分，TTFT 从约 1.4 s 降到约 0.2 s（I10）。下面的 L2 校验本身也起到预热作用；J2 会把预热做成启动脚本。

## 3. 接入 Agent 并校验

在 `.env` 中设置：

```
LLAMA_SERVER_URL=http://127.0.0.1:8080/v1
```

在 Windows 上要写 `127.0.0.1`，不要写 `localhost`：`localhost` 会先尝试 `::1`，WSL 的端口转发不应答它，部分客户端
要等约 2 s 才回落到 IPv4。Docker Compose 中保持默认的 `http://llama-server:8080/v1`。

验收（L2 主集，同类机器上应为 56/60 = 93.3%）：

```bash
uv run --env-file .env python eval/runners/run_slot_eval.py --extractor local --dataset main --concurrency 1
```

`config/settings.yaml` 中 `local_slot` 的超时默认是 5 s。如果在较慢的 CPU 上头几个请求超时，把
`LOCAL_SLOT_TIMEOUT_S` 调大。
