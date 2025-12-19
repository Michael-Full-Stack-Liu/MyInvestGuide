-- =============================================================================
-- PostgreSQL 初始化脚本
-- =============================================================================
-- 此脚本仅在容器首次启动时执行
-- 用于创建 MLflow 和其他服务所需的数据库和用户
-- =============================================================================

-- 显示当前连接信息
\echo '=================================================='
\echo '正在初始化 Congress Trading MLOps 数据库...'
\echo '=================================================='

-- -----------------------------------------------------------------------------
-- 1. 切换到 mlflow_db 数据库
-- -----------------------------------------------------------------------------
-- 注意: 主数据库 (mlflow_db) 已通过环境变量 POSTGRES_DB 自动创建
\c mlflow_db

-- 启用 UUID 扩展 (某些 MLflow 版本可能需要)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 2. 预测日志表 (Prediction Logs)
-- =============================================================================
-- 存储 API 预测请求和结果，用于：
--   - Drift 检测 (与训练数据比较)
--   - 模型性能监控
--   - 审计追踪

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    
    -- 输入数据 (用于 drift 检测)
    politician_name VARCHAR(100) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    trade_type VARCHAR(20) NOT NULL,
    amount_min DECIMAL(15, 2),
    filed_after INTEGER,
    party VARCHAR(50),
    chamber VARCHAR(50),
    transaction_date DATE,
    
    -- 预测结果
    prediction INTEGER NOT NULL,
    label VARCHAR(50) NOT NULL,
    recommendation VARCHAR(20) NOT NULL,
    
    -- 模型信息
    model_version VARCHAR(50),
    model_type VARCHAR(20),  -- 'mlflow' or 'autogluon'
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_politician ON predictions(politician_name);

-- =============================================================================
-- 3. Drift 检测历史表 (Drift History)
-- =============================================================================
-- 存储每次 drift 检测的结果，用于：
--   - 监控数据分布变化趋势
--   - 触发告警历史记录
--   - 模型是否需要重新训练的决策依据

CREATE TABLE IF NOT EXISTS drift_history (
    id SERIAL PRIMARY KEY,
    
    -- Drift 检测结果
    drift_share DECIMAL(5, 4),  -- 0.0000 - 1.0000
    is_drifted BOOLEAN NOT NULL,
    drifted_columns TEXT[],  -- PostgreSQL 数组类型
    total_columns INTEGER,
    
    -- 告警状态
    threshold DECIMAL(5, 4),
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_level VARCHAR(20),  -- 'info', 'warn', 'error'
    
    -- 检测方式
    check_type VARCHAR(20) NOT NULL,  -- 'scheduled' or 'manual'
    reference_count INTEGER,  -- 参考数据行数
    current_count INTEGER,    -- 当前数据行数
    
    -- 元数据
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_drift_history_checked_at ON drift_history(checked_at);
CREATE INDEX IF NOT EXISTS idx_drift_history_is_drifted ON drift_history(is_drifted);

-- =============================================================================
-- 4. 模型回测结果表 (Backtest Results)
-- =============================================================================
-- 存储模型回测结果，用于：
--   - 追踪模型随时间的表现
--   - 比较不同模型版本
--   - 支持模型选择决策

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    
    -- 模型信息
    model_version VARCHAR(100) NOT NULL,
    mlflow_run_id VARCHAR(100),
    model_type VARCHAR(50),  -- 'autogluon', 'xgboost', etc.
    
    -- 回测配置
    backtest_start_date DATE,
    backtest_end_date DATE,
    train_size INTEGER,
    test_size INTEGER,
    
    -- 分类指标
    accuracy DECIMAL(5, 4),
    precision_macro DECIMAL(5, 4),
    recall_macro DECIMAL(5, 4),
    f1_macro DECIMAL(5, 4),
    roc_auc_macro DECIMAL(5, 4),
    
    -- 业务指标 (Alpha-based)
    avg_alpha DECIMAL(8, 4),
    std_alpha DECIMAL(8, 4),
    information_ratio DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    total_trades INTEGER,
    
    -- 高置信度交易指标
    high_conviction_trades INTEGER,
    high_conviction_avg_alpha DECIMAL(8, 4),
    high_conviction_win_rate DECIMAL(5, 4),
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_backtest_model_version ON backtest_results(model_version);
CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON backtest_results(created_at);

-- =============================================================================
-- 5. 创建视图 (方便查询)
-- =============================================================================

-- 最近7天预测统计
CREATE OR REPLACE VIEW v_prediction_stats_7d AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN recommendation = 'FOLLOW' THEN 1 ELSE 0 END) as follow_count,
    SUM(CASE WHEN recommendation = 'SKIP' THEN 1 ELSE 0 END) as skip_count,
    AVG(prediction) as avg_prediction
FROM predictions
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Drift 趋势视图
CREATE OR REPLACE VIEW v_drift_trend AS
SELECT 
    DATE(checked_at) as date,
    AVG(drift_share) as avg_drift_share,
    MAX(drift_share) as max_drift_share,
    SUM(CASE WHEN alert_sent THEN 1 ELSE 0 END) as alerts_sent
FROM drift_history
WHERE checked_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(checked_at)
ORDER BY date DESC;

-- =============================================================================
-- 6. 完成
-- =============================================================================
\echo '=================================================='
\echo '数据库初始化完成!'
\echo '已创建数据库: mlflow_db'
\echo '已创建表:'
\echo '  - predictions (预测日志)'
\echo '  - drift_history (Drift 检测历史)'
\echo '  - backtest_results (回测结果)'
\echo '已创建视图:'
\echo '  - v_prediction_stats_7d'
\echo '  - v_drift_trend'
\echo '=================================================='
