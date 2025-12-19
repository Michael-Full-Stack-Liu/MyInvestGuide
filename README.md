# 🤖 Congress Trading ML Prediction Service

> 国会议员股票交易质量预测服务 - 完整的 MLOps 端到端解决方案

基于 AutoGluon 的机器学习模型，预测国会议员股票交易的潜在收益，并提供 RESTful API 服务。

---

## 📋 目录

- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [API 文档](#-api-文档)
- [模型训练](#-模型训练)
- [漂移检测与报警](#-漂移检测与报警)
- [配置说明](#-配置说明)
- [项目结构](#-项目结构)

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 🎯 **AutoGluon 模型** | 自动特征工程 + 模型选择 + 超参数优化 |
| 🔄 **DVC 管道** | 可复现的数据处理和模型训练流程 |
| 📊 **MLflow 追踪** | 实验追踪、指标记录、模型版本管理 |
| 🚀 **FastAPI 服务** | 高性能 RESTful 预测 API |
| 📉 **漂移检测** | 基于 Evidently 的数据分布监控 |
| 📱 **多渠道报警** | Telegram / Email / Discord 通知 |
| 🗄️ **PostgreSQL** | 存储预测日志和漂移历史 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│               Congress Trading ML Prediction                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DVC Pipeline                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │load_data │─▶│clean_data│─▶│ feature  │─▶│  train   │ │   │
│  │  │          │  │          │  │ engineer │  │ (AutoGL) │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   FastAPI Service                        │   │
│  │  /predict  /predict/batch  /drift/check  /health        │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  PostgreSQL │  │   MLflow    │  │  Telegram   │             │
│  │ (预测日志)   │  │ (实验追踪)  │  │   (报警)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 前置要求

- Docker Desktop
- (可选) Telegram Bot Token (用于报警)

### 1. 进入项目目录

```bash
cd d:\Front-end-project\congress_new_predict
```

### 2. 配置环境变量

复制并编辑 `.env` 文件：

```bash
cp .env.example .env
```

最小配置：
```bash
# PostgreSQL
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=mlflow_db

# Telegram 报警 (推荐)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. 启动所有服务

```bash
docker compose up -d postgres mlflow api
```

### 4. 验证服务

```bash
# 检查健康状态
curl http://localhost:8000/health

# 查看 API 文档
# 访问 http://localhost:8000/docs
```

---

## 📡 API 文档

### 基础信息

| 项目 | 值 |
|------|-----|
| Base URL | `http://localhost:8000` |
| API 文档 | `http://localhost:8000/docs` |
| OpenAPI | `http://localhost:8000/openapi.json` |

### 端点列表

#### 健康检查
```http
GET /health
```

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "autogluon",
  "database": "connected",
  "version": "2.1.0"
}
```

#### 单条预测
```http
POST /predict
Content-Type: application/json

{
  "politician_name": "Nancy Pelosi",
  "ticker": "AAPL",
  "type": "Purchase",
  "amount_min": 100000,
  "filed_after": 30
}
```

**响应示例：**
```json
{
  "prediction": 2,
  "label": "Good",
  "recommendation": "FOLLOW"
}
```

#### 批量预测
```http
POST /predict/batch
Content-Type: application/json

{
  "trades": [
    {"politician_name": "Nancy Pelosi", "ticker": "AAPL", "type": "Purchase", "amount_min": 100000},
    {"politician_name": "Dan Crenshaw", "ticker": "MSFT", "type": "Sale", "amount_min": 50000}
  ]
}
```

**响应示例：**
```json
{
  "predictions": [
    {"prediction": 2, "label": "Good", "recommendation": "FOLLOW"},
    {"prediction": 0, "label": "Weak", "recommendation": "SKIP"}
  ],
  "follow_count": 1,
  "skip_count": 1
}
```

#### 手动触发漂移检测
```http
POST /drift/check
```

**响应示例：**
```json
{
  "drift_share": 0.82,
  "feature_drift": 1.0,
  "prediction_drift": 0.4,
  "is_drifted": true,
  "drifted_columns": ["amount_min", "filed_after"],
  "prediction_distribution": "Class0:60.0% | Class1:20.0% | Class2:15.0% | Class3:5.0%",
  "alert_sent": true,
  "reference_count": 35778,
  "current_count": 100
}
```

**字段说明：**
| 字段 | 说明 |
|------|------|
| `drift_share` | 综合漂移得分 (70% 特征漂移 + 30% 预测分布漂移) |
| `feature_drift` | 输入特征分布漂移 (使用 Evidently) |
| `prediction_drift` | 预测结果分布漂移 (与预期分布对比) |
| `prediction_distribution` | 当前预测分布 (Class0~Class3) |
| `alert_sent` | 是否发送了 Telegram 通知 |

> **注意**: 无论是否检测到漂移，都会发送 Telegram 通知。正常情况发送简单报告，异常情况发送详细报告。

#### 获取漂移历史
```http
GET /drift/history?days=7
```

#### 获取漂移趋势
```http
GET /drift/trend?days=30
```

#### 获取预测统计
```http
GET /predictions/stats?days=7
```

#### 获取模型信息
```http
GET /model/info
```

**响应示例：**
```json
{
  "loaded": true,
  "type": "autogluon",
  "version": "v1.0",
  "model_path": "/app/models/autogluon"
}
```

---

### 🆕 预测准确率验证 (后验数据验证)

用于验证模型预测与实际结果的准确率。支持原始 CSV 格式和处理后的 Parquet 文件。

#### 验证准确率
```http
POST /validate/accuracy?file_path=/app/data/congress_trading_2025-12-13.csv
```

**参数说明：**
| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `file_path` | ✅ | - | CSV/Parquet 文件路径 (容器内路径) |
| `alpha_column` | ❌ | `Alpha_180` | Alpha 值列名 |
| `min_records` | ❌ | `100` | 最小记录数 |

**支持的文件格式：**

| 文件类型 | 说明 | Alpha 计算 |
|----------|------|------------|
| **原始 CSV** (如 `congress_trading_2025-12-13.csv`) | 包含价格列 | 自动计算 Alpha_180 |
| **处理后的 Parquet** (如 `01_raw_trades.parquet`) | 已包含 Alpha_180 | 直接使用 |

**原始 CSV 必需列：**
- `Politician Name`, `Ticker`, `Type`, `Amount Min`, `Filed After` (用于预测)
- `Entry Price`, `Exit Price 180` (用于计算股票收益)
- `SPY Entry`, `SPY Exit 180` (可选，用于计算超额收益)

> **注意**: 如果 CSV 中没有 `Alpha_180` 列，系统会自动从价格列计算：
> - `Stock_Return = (Exit Price 180 - Entry Price) / Entry Price`
> - `Alpha_180 = Stock_Return - SPY_Return` (如果有 SPY 数据)

**响应示例：**
```json
{
  "total_records": 5000,
  "matched_records": 5000,
  "accuracy": 0.42,
  "precision_by_class": {
    "class_0": 0.65,
    "class_1": 0.38,
    "class_2": 0.45,
    "class_3": 0.52
  },
  "recall_by_class": {
    "class_0": 0.72,
    "class_1": 0.31,
    "class_2": 0.48,
    "class_3": 0.35
  },
  "confusion_matrix": {
    "actual_0": {"pred_0": 1200, "pred_1": 300, "pred_2": 100, "pred_3": 50},
    "actual_1": {"pred_0": 400, "pred_1": 600, "pred_2": 200, "pred_3": 100},
    "..."
  },
  "class_distribution_actual": {
    "class_0": 0.55,
    "class_1": 0.25,
    "class_2": 0.15,
    "class_3": 0.05
  },
  "class_distribution_predicted": {
    "class_0": 0.60,
    "class_1": 0.22,
    "class_2": 0.13,
    "class_3": 0.05
  },
  "follow_accuracy": 0.68,
  "recommendations": [
    "📊 准确率一般，考虑调整特征工程",
    "✅ FOLLOW 推荐可信度高"
  ]
}
```

**使用示例 (PowerShell)：**
```powershell
# 1. 将 CSV 文件复制到容器可访问的目录
docker cp congress_trading_2025.csv congress_api:/app/data/

# 2. 调用验证 API
Invoke-RestMethod -Uri "http://localhost:8000/validate/accuracy?file_path=/app/data/congress_trading_2025.csv" -Method Post | ConvertTo-Json -Depth 5
```

**使用示例 (curl)：**
```bash
# 1. 复制文件
docker cp congress_trading_2025.csv congress_api:/app/data/

# 2. 调用 API
curl -X POST "http://localhost:8000/validate/accuracy?file_path=/app/data/congress_trading_2025.csv"
```

---

## 🎓 模型训练

### DVC 管道

项目使用 DVC (Data Version Control) 管理 ML 管道：

```
load_data → clean_data → engineer_features → train_model
```

### 运行训练

```bash
# 使用 Docker
docker compose run train

# 或运行特定阶段
docker compose run train dvc repro train_model
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `time_limit` | 7200 | 训练时间限制(秒) |
| `preset` | best_quality | AutoGluon 预设 |

### 特征工程

模型使用 **78 个特征**，包括：
- 基础特征 (政客、金额、延迟等)
- 时间特征 (月/周/季度)
- 历史行为特征 (交易频率、平均金额)
- 交互特征 (政党 × 交易类型)
- 滞后特征 (历史统计)

### 预测目标

基于 **180天超额收益 (Alpha)** 的 4 分类：

| 类别 | Alpha 范围 | 说明 |
|------|-----------|------|
| 0 - Weak | < 0% | 亏损 |
| 1 - Fair | 0% ~ 10% | 微利 |
| 2 - Good | 10% ~ 20% | 良好 |
| 3 - Excellent | > 20% | 优秀 |

---

## 📉 漂移检测与报警

### 综合漂移检测

系统执行两种漂移检测：

| 检测类型 | 权重 | 说明 |
|----------|------|------|
| **特征漂移** | 70% | 使用 Evidently 检测输入特征分布变化 |
| **预测分布漂移** | 30% | 检测预测结果分布与训练时的偏差 |

**综合得分** = 特征漂移 × 0.7 + 预测分布漂移 × 0.3

### 通知机制

**无论是否检测到漂移，都会发送 Telegram 通知：**

| 情况 | 通知类型 | 内容 |
|------|----------|------|
| ✅ 正常 (综合得分 ≤ 30%) | 简单报告 | 检测时间、得分、分布概览 |
| ⚠️ 异常 (综合得分 > 30%) | 详细报告 | 完整分析、漂移特征、建议 |
| 🚨 错误 | 错误通知 | 错误信息和时间 |

### 通知示例

**正常情况 (简单报告)：**
```
✅ 漂移检测报告 - 正常
━━━━━━━━━━━━━━━━━━━━━━━
⏰ 检测时间: 2025-12-18 22:40
📊 检测类型: scheduled

📈 特征漂移: 5.0%
📉 预测分布漂移: 10.0%
✅ 综合得分: 6.5% (阈值: 30%)

📋 数据: 35,778 参考 / 100 当前
📊 分布: Class0:60.0% | Class1:20.0% | Class2:15.0% | Class3:5.0%

状态: 一切正常 ✓
```

**异常情况 (详细报告)：**
```
🔍 漂移检测报告 - 检测到异常
━━━━━━━━━━━━━━━━━━━━━━━
⏰ 检测时间: 2025-12-18 22:40
📊 检测类型: manual

📈 特征漂移: 100.0%
📉 预测分布漂移: 40.0%
⚠️ 综合得分: 82.0% (阈值: 30%)

🔴 漂移特征: amount_min, filed_after, party

📊 当前预测分布:
Class0:100.0% | Class1:0.0% | Class2:0.0% | Class3:0.0%

📋 数据规模:
• 参考数据: 35,778 条
• 当前数据: 15 条

💡 建议: 考虑重新训练模型
```

### 工作原理

1. **定时检测**: 每 7 天自动检查数据漂移 (可配置)
2. **手动检测**: 调用 `POST /drift/check`
3. **阈值判断**: 综合得分 > 30% 触发警报
4. **多渠道通知**: Telegram (推荐) / Email / Discord

### 支持的报警渠道

| 渠道 | 配置 | 免费 |
|------|------|------|
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | ✅ |
| SMTP Email | `SMTP_*` 系列变量 | ✅ (Gmail) |
| Discord | `DISCORD_WEBHOOK_URL` | ✅ |
| W&B | `WANDB_API_KEY` | ❌ (需付费) |

### 配置 Telegram 报警

```bash
# .env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 测试报警

```bash
docker exec congress_api python -c "
from src.monitoring.alert import send_alert
send_alert('🧪 测试报警', '这是测试消息', level='info')
"
```

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| **数据库** |||
| `POSTGRES_USER` | `mlflow` | 数据库用户 |
| `POSTGRES_PASSWORD` | - | 数据库密码 |
| `POSTGRES_DB` | `mlflow_db` | 数据库名 |
| **模型** |||
| `MODEL_PATH` | `/app/models/autogluon` | AutoGluon 模型路径 |
| `MLFLOW_MODEL_PATH` | `/app/models/mlflow_model` | MLflow 模型路径 |
| **漂移检测** |||
| `DRIFT_CHECK_INTERVAL_DAYS` | `7` | 自动检测间隔 |
| `DRIFT_THRESHOLD` | `0.3` | 报警阈值 (30%) |
| **报警** |||
| `TELEGRAM_BOT_TOKEN` | - | Telegram Bot Token |
| `TELEGRAM_CHAT_ID` | - | Telegram Chat ID |
| `WANDB_ALERTS_ENABLED` | `false` | W&B 报警开关 |

---

## 📁 项目结构

```
congress_new_predict/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI 应用
│   ├── model/
│   │   ├── data_loader.py      # 数据加载
│   │   ├── data_cleaner.py     # 数据清洗
│   │   ├── feature_engineer.py # 特征工程
│   │   ├── trainer_autogluon.py# AutoGluon 训练
│   │   └── mlflow_wrapper.py   # MLflow 模型封装
│   ├── monitoring/
│   │   ├── drift.py            # 漂移检测 (Evidently)
│   │   └── alert.py            # 多渠道报警
│   └── database.py             # 数据库操作
├── models/
│   └── autogluon/              # 训练好的模型
├── data/
│   └── intermediate/           # 中间数据文件
├── dvc.yaml                    # DVC 管道定义
├── docker-compose.yml          # Docker 服务编排
├── Dockerfile.api              # API 服务镜像
├── Dockerfile.train            # 训练服务镜像
├── requirements.txt            # Python 依赖
└── .env.example                # 环境变量模板
```

---

## 🔧 常用命令

```bash
# 启动服务
docker compose up -d postgres mlflow api

# 查看日志
docker compose logs -f api

# 停止服务
docker compose down

# 重建 API 服务
docker compose build api
docker compose up -d api --force-recreate

# 运行模型训练
docker compose run train

# 进入 API 容器
docker compose exec api bash

# 检查数据库
docker compose exec postgres psql -U mlflow -d mlflow_db

# 测试预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"politician_name":"Nancy Pelosi","ticker":"AAPL","type":"Purchase","amount_min":100000}'
```

---

## 🌐 服务端口

| 服务 | 端口 | 用途 |
|------|------|------|
| API | 8000 | 预测服务 |
| MLflow | 5000 | 实验追踪 UI |
| PostgreSQL | 5432 | 数据库 |

---

## 🔗 与 congress_new 集成

本服务设计为 [congress_new](../congress_new) 的后端预测服务：

1. 两个项目通过 Docker 网络 `congress_mlops_network` 通信
2. congress_new 调用 `/predict/batch` 获取预测
3. 仅推送 prediction=2,3 的高质量交易

### 启动顺序

```bash
# 1. 先启动本服务
cd d:\Front-end-project\congress_new_predict
docker compose up -d

# 2. 再启动通知服务
cd d:\Front-end-project\congress_new
docker compose up -d
```

---

## ❓ 常见问题

### Q: 模型加载失败?

确保 `models/autogluon/` 目录存在且包含训练好的模型。

### Q: 漂移检测始终返回 0%?

已修复 Evidently 0.7.x API 兼容性问题。确保使用最新代码。

### Q: Telegram 报警不工作?

检查环境变量是否正确设置：
```bash
docker exec congress_api env | grep TELEGRAM
```

### Q: 如何重新训练模型?

```bash
docker compose run train dvc repro --force
```

---

## 📄 License

MIT License

---

## 🔗 相关项目

- [congress_new](../congress_new) - 数据抓取与通知服务
