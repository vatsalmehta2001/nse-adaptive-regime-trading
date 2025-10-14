# Trading Strategy Overview

## Core Strategy: Adaptive Regime Trading

The core strategy combines regime detection with adaptive model selection to optimize performance across different market conditions.

## Strategy Components

### 1. Market Regime Detection

**Purpose**: Identify current market state to adapt trading approach

**Regimes Identified**:
- **Bull Market**: Strong uptrend, high momentum
- **Bear Market**: Strong downtrend, risk-off
- **Sideways Market**: Range-bound, mean-reverting
- **High Volatility**: Large price swings, uncertainty

**Detection Methods**:
- Wasserstein distance between return distributions
- Hidden Markov Models for regime probabilities
- GARCH models for volatility clustering
- Trend strength indicators

### 2. Alpha Factor Generation

**Qlib Alpha-158 Library**:
- Price-based features (OHLC patterns)
- Volume-based features (volume trends)
- Price trends (moving averages)
- Volatility features (standard deviation)
- Momentum indicators (RSI, MACD)
- Cross-sectional features (relative strength)

**Custom Features**:
- Sector rotation signals
- Market breadth indicators
- Liquidity measures
- Order flow imbalance

### 3. Predictive Models

**Qlib Models**:
- **LightGBM**: Primary model for return prediction
- **XGBoost**: Alternative gradient boosting
- **CatBoost**: Handles categorical features

**Model Outputs**:
- Expected returns
- Return confidence intervals
- Feature importance scores

### 4. Reinforcement Learning Layer

**RL Agents**:
- **PPO (Proximal Policy Optimization)**: Primary agent
- **A2C (Advantage Actor-Critic)**: Alternative
- **SAC (Soft Actor-Critic)**: For continuous actions

**State Space**:
- Current positions
- Portfolio metrics
- Market features
- Regime indicators

**Action Space**:
- Hold (no action)
- Buy (increase position)
- Sell (decrease position)
- Position size adjustments

**Reward Function**:
- Risk-adjusted returns (Sharpe ratio)
- Penalty for excessive trading
- Drawdown penalties

### 5. Portfolio Construction

**Position Sizing**:
- Risk-based sizing (1-2% risk per trade)
- Volatility-weighted allocation
- Kelly criterion for optimal sizing

**Diversification**:
- Maximum 20 positions
- Sector limits (30% per sector)
- Correlation constraints

**Rebalancing**:
- Daily position review
- Threshold-based rebalancing (5% drift)
- Regime-triggered rebalancing

### 6. Risk Management

**Position-Level Risk**:
- Stop loss: 2-3% fixed or trailing
- Take profit: Multiple targets (2%, 5%, 10%)
- Maximum holding period: 30 days

**Portfolio-Level Risk**:
- Maximum drawdown: 15%
- Daily loss limit: 3%
- VaR limit: 2% at 95% confidence

**Emergency Controls**:
- Kill switch for consecutive losses
- Circuit breaker for flash crashes
- Position flattening during high volatility

## Strategy Workflow

### 1. Pre-Market (8:00-9:15 AM IST)

1. Download overnight data
2. Update feature calculations
3. Detect current market regime
4. Train/update models (if scheduled)
5. Generate universe of tradable stocks
6. Calculate position targets

### 2. Market Open (9:15 AM)

1. Execute opening orders
2. Monitor order fills
3. Adjust unfilled orders

### 3. Intraday (9:30 AM - 3:15 PM)

1. Monitor positions every 5 minutes
2. Check stop loss and take profit levels
3. Rebalance if threshold breached
4. Execute intraday trades (if regime changes)
5. Monitor risk metrics continuously

### 4. Market Close (3:30 PM)

1. Close MIS positions (if any)
2. Review day's performance
3. Update portfolio metrics
4. Log trades and positions

### 5. Post-Market (4:00 PM - 8:00 PM)

1. Calculate daily performance
2. Update models with new data
3. Plan next day's trades
4. Generate daily reports

## Signal Generation

### Entry Signals

**Long Entry**:
- Qlib prediction > 0.5 confidence
- Positive RL agent recommendation
- Regime = Bull or Sideways
- No existing position or below max size
- Stock passes liquidity filter

**Short Entry** (if enabled):
- Qlib prediction < -0.5 confidence
- Negative RL agent recommendation
- Regime = Bear
- Borrowing available

### Exit Signals

**Standard Exit**:
- Stop loss hit (2-3% loss)
- Take profit hit (2%, 5%, or 10%)
- Regime change to unfavorable
- Holding period exceeded

**Emergency Exit**:
- Portfolio drawdown > 15%
- Daily loss > 3%
- Multiple consecutive losses
- System error or API disconnect

## Performance Expectations

### Backtested Metrics (Example Targets)

- **Annual Return**: 15-25%
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 15%
- **Win Rate**: 50-60%
- **Profit Factor**: > 1.5

### Risk Metrics

- **Daily VaR (95%)**: < 2%
- **Beta to Nifty 50**: 0.8-1.2
- **Average Trade Duration**: 5-10 days
- **Turnover**: 10-20x annually

## Strategy Variants

### Conservative

- Lower position sizes (3-5%)
- Tighter stops (1.5%)
- Longer holding periods
- Higher quality filters

### Aggressive

- Larger position sizes (7-10%)
- Wider stops (4-5%)
- More frequent trading
- Include mid-cap stocks

### Market Neutral

- Equal long and short positions
- Beta-hedged portfolio
- Lower overall exposure
- Focus on alpha generation

## Known Limitations

1. **Transaction Costs**: Impact on high-frequency rebalancing
2. **Slippage**: Market impact for large orders
3. **Regime Detection Lag**: Delayed regime identification
4. **Model Drift**: Performance degradation over time
5. **Black Swan Events**: Unpredictable market shocks

## Continuous Improvement

### Model Retraining

- Weekly: Update with latest week's data
- Monthly: Full model retraining
- Quarterly: Hyperparameter optimization

### Strategy Evolution

- Monitor performance metrics
- A/B test strategy variants
- Incorporate new features
- Adapt to market changes

## References

- Qlib documentation on factor investing
- Reinforcement learning for trading papers
- NSE trading rules and regulations
- Risk management best practices

