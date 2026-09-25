# 在 Claude Desktop 中使用 booking-mcp

booking-mcp 通过 stdio 暴露 5 个预约工具，可直接挂进 Claude Desktop / Cursor 等 MCP 客户端，
不经过本项目的 Agent。

| Tool | 作用 |
|---|---|
| `find_companions` | 按模式、时间、时长 + 段位/位置/性别/开麦/预算/风格搜索可约陪玩师；零结果时按 时间±1h → 性别 → 位置 放宽（预算与段位不放宽） |
| `quote_price` | 单价 × 服务系数 × 时长 |
| `create_booking` | 下单（`pending_payment`），事务内二次检测档期冲突 |
| `list_my_bookings` | 我的订单 |
| `cancel_booking` | `dry_run=true` 报退款；`dry_run=false` 取消并释放档期 |

## 1. 准备数据库

```bash
uv sync
uv run python scripts/seed_all.py --reset
```

种子数据包含 30 名陪玩师、未来 14 天档期，以及演示用户 `demo`（`user_id = 1`）。
也可以跳过这一步，在启动参数里加 `--seed-if-empty`，首次启动时自动生成。

## 2. 配置 Claude Desktop

编辑配置文件（Windows：`%APPDATA%\Claude\claude_desktop_config.json`；
macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`），把路径换成本仓库位置：

```json
{
  "mcpServers": {
    "rift-booking": {
      "command": "C:\\path\\to\\rift-companion-agent\\.venv\\Scripts\\python.exe",
      "args": [
        "-m", "booking_mcp",
        "--transport", "stdio",
        "--db", "C:\\path\\to\\rift-companion-agent\\data\\booking.db",
        "--seed-if-empty"
      ]
    }
  }
}
```

使用 uv 的等价写法（macOS / Linux 路径示例）：

```json
{
  "mcpServers": {
    "rift-booking": {
      "command": "uv",
      "args": ["--directory", "/path/to/rift-companion-agent", "run", "booking-mcp", "--seed-if-empty"]
    }
  }
}
```

可选参数：

| 参数 | 说明 |
|---|---|
| `--now 2026-10-01T14:00` | 固定"当前时间"，演示与评测可复现（种子档期从该日期开始） |
| `--embedding mock` / `settings` | 启用 `style_preference` 风格相似度排序（`settings` 读取 `config/settings.yaml` 的 embedding 配置） |
| `--trace-db data/traces.db` | 把工具调用 span 写入 SQLite |

重启 Claude Desktop 后，工具列表中应出现 `rift-booking` 的 5 个工具。

## 3. 演示对话

> 我是 user_id 1。帮我找明晚 8 点的大乱斗陪玩，两小时，想要温柔一点的。
>
> 就第一个吧，报个价然后下单。
>
> 我现在有哪些订单？如果现在取消能退多少？

Claude 会依次调用 `find_companions` → `quote_price` → `create_booking` → `list_my_bookings`
→ `cancel_booking(dry_run=true)`。截图见 `docs/img/mcp-desktop-01-running.png` ～ `mcp-desktop-04-booking.png`。

## 4. 不开 Claude Desktop 的自检

`scripts/mcp_smoke.py` 以与 Claude Desktop 相同的方式（stdio 子进程）跑完 找人 → 报价 → 下单 → 查订单 → 退款试算：

```bash
uv run python scripts/mcp_smoke.py
```

## 错误码

业务错误以工具错误返回（`isError: true`），结构化内容为：

```json
{"error": {"code": "SLOT_CONFLICT", "jsonrpc_code": -32001, "message": "companion 3 is not available ..."}}
```

| code | jsonrpc_code | 含义 |
|---|---|---|
| `SLOT_CONFLICT` | -32001 | 该时段已被占用（并发下单时只有一个成功） |
| `NOT_FOUND` | -32002 | 陪玩师 / 用户 / 订单 / 工具不存在 |
| `FORBIDDEN` | -32003 | 操作他人订单 |
| `INVALID_ARGUMENT` | -32602 | 参数不合法（枚举越界、时长非 0.5 步长、时间已过、陪玩师不接该模式等） |
| `INTERNAL` | -32603 | 未预期异常（详情只写服务端 stderr 日志） |

## HTTP 模式

Agent 使用 Streamable HTTP（无状态，JSON 响应）：

```bash
uv run booking-mcp --transport http --port 8101
# MCP 端点：http://127.0.0.1:8101/mcp
```
