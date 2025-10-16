"""
Vectorized Backtesting Engine.

Efficient backtesting with transaction costs, position tracking, and regime awareness.
Fully vectorized for speed on large datasets.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Features:
    - Fully vectorized (fast)
    - Transaction costs (commission + slippage)
    - Position tracking
    - Daily portfolio statistics
    - Regime-aware strategy support
    - Rebalancing control
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital in INR
            commission: Per-trade commission (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        logger.info(f"Initialized backtest engine: capital={initial_capital:,.0f}, "
                   f"commission={commission:.3%}, slippage={slippage:.3%}")

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        rebalance_freq: str = "daily"
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.

        Args:
            signals: DataFrame with [date, symbol, signal, weight]
            prices: DataFrame with [date, symbol, close] or multi-index
            regime_labels: Optional regime per date
            rebalance_freq: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame with daily portfolio metrics
        """
        logger.info(f"Running backtest with {rebalance_freq} rebalancing...")

        # Prepare data
        signals = signals.copy()

        # Ensure date column
        if 'date' in signals.columns:
            signals['date'] = pd.to_datetime(signals['date'])
        else:
            raise ValueError("Signals must have 'date' column")

        # Prepare prices
        if isinstance(prices.index, pd.MultiIndex):
            # Multi-index (symbol, date) - pivot to wide format
            prices_wide = prices['close'].unstack(level='symbol')
        elif 'date' in prices.columns and 'symbol' in prices.columns:
            # Long format - pivot
            prices_wide = prices.pivot(index='date', columns='symbol', values='close')
        else:
            # Already wide format
            prices_wide = prices.copy()

        prices_wide.index = pd.to_datetime(prices_wide.index)

        # Get unique dates from signals
        dates = sorted(signals['date'].unique())

        # Initialize results
        portfolio_values = []
        cash = self.initial_capital
        positions = pd.Series(0.0, index=prices_wide.columns)  # Shares held

        # Track previous weights for turnover
        prev_weights = pd.Series(0.0, index=prices_wide.columns)

        for i, date in enumerate(dates):
            # Get current prices
            if date not in prices_wide.index:
                logger.warning(f"No price data for {date}, skipping")
                continue

            current_prices = prices_wide.loc[date].dropna()

            # Calculate current portfolio value
            position_values = positions * current_prices
            portfolio_value = cash + position_values.sum()

            # Determine if rebalancing day
            should_rebalance = self._should_rebalance(date, i, rebalance_freq)

            if should_rebalance:
                # Get signals for this date
                date_signals = signals[signals['date'] == date].copy()

                if len(date_signals) > 0:
                    # Calculate target positions
                    target_weights = pd.Series(0.0, index=prices_wide.columns)

                    for _, row in date_signals.iterrows():
                        symbol = row['symbol']
                        weight = row['weight']

                        if symbol in target_weights.index:
                            target_weights[symbol] = weight

                    # Calculate current weights
                    current_weights = position_values / portfolio_value

                    # Calculate trades needed
                    weight_changes = target_weights - current_weights

                    # Calculate turnover
                    turnover = weight_changes.abs().sum()

                    # Execute trades
                    for symbol in weight_changes[weight_changes != 0].index:
                        if symbol not in current_prices.index:
                            continue

                        weight_change = weight_changes[symbol]
                        target_value = portfolio_value * target_weights[symbol]
                        current_value = position_values.get(symbol, 0)

                        trade_value = target_value - current_value
                        price = current_prices[symbol]

                        # Apply slippage
                        if trade_value > 0:  # Buy
                            execution_price = price * (1 + self.slippage)
                        else:  # Sell
                            execution_price = price * (1 - self.slippage)

                        # Calculate shares to trade
                        shares_to_trade = trade_value / execution_price

                        # Execute trade
                        positions[symbol] += shares_to_trade

                        # Update cash (with commission)
                        trade_cost = abs(trade_value) * self.commission
                        cash -= (shares_to_trade * execution_price + trade_cost)

                    # Store turnover
                    prev_weights = target_weights.copy()
                else:
                    turnover = 0.0
            else:
                turnover = 0.0

            # Recalculate portfolio value after trades
            position_values = positions * current_prices
            portfolio_value = cash + position_values.sum()

            # Calculate metrics
            n_positions = (positions.abs() > 1e-6).sum()
            invested_value = position_values.sum()

            # Get regime if available
            regime = None
            if regime_labels is not None and date in regime_labels.index:
                regime = regime_labels[date]

            # Store results
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'invested_value': invested_value,
                'positions_count': n_positions,
                'turnover': turnover,
                'regime': regime
            })

        # Create results DataFrame
        results = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate returns
        results['daily_return'] = results['portfolio_value'].pct_change()
        results['cumulative_return'] = (
            results['portfolio_value'] / self.initial_capital - 1
        )

        # Fill first day
        results.loc[results.index[0], 'daily_return'] = 0.0
        results.loc[results.index[0], 'cumulative_return'] = 0.0

        logger.info(f"Backtest complete: {len(results)} days")
        logger.info(f"Final value: {results['portfolio_value'].iloc[-1]:,.0f}")
        logger.info(f"Total return: {results['cumulative_return'].iloc[-1]:.2%}")

        return results

    def _should_rebalance(
        self,
        date: pd.Timestamp,
        day_index: int,
        rebalance_freq: str
    ) -> bool:
        """
        Determine if portfolio should rebalance on this date.

        Args:
            date: Current date
            day_index: Index in the date sequence
            rebalance_freq: Rebalancing frequency

        Returns:
            True if should rebalance
        """
        if rebalance_freq == "daily":
            return True
        elif rebalance_freq == "weekly":
            # Rebalance on Monday (weekday = 0)
            return date.weekday() == 0
        elif rebalance_freq == "monthly":
            # Rebalance on first trading day of month
            return date.day <= 7 and (day_index == 0 or date.day == 1)
        else:
            return True

    def calculate_positions(
        self,
        signals: pd.DataFrame,
        capital: float,
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Convert signals to actual positions.

        Args:
            signals: DataFrame with [symbol, signal, weight]
            capital: Available capital
            prices: Current prices per symbol

        Returns:
            DataFrame with [symbol, shares, value, weight]
        """
        positions = []

        for _, row in signals.iterrows():
            symbol = row['symbol']
            weight = row['weight']

            if symbol in prices.index and weight != 0:
                target_value = capital * weight
                price = prices[symbol]
                shares = target_value / price

                positions.append({
                    'symbol': symbol,
                    'shares': shares,
                    'value': target_value,
                    'weight': weight,
                    'price': price
                })

        return pd.DataFrame(positions)

    def apply_transaction_costs(
        self,
        trades: pd.DataFrame,
        prices: pd.Series
    ) -> float:
        """
        Calculate transaction costs.

        Args:
            trades: DataFrame with [symbol, shares] to trade
            prices: Current prices

        Returns:
            Total cost (negative number)
        """
        total_cost = 0.0

        for _, row in trades.iterrows():
            symbol = row['symbol']
            shares = row['shares']

            if symbol in prices.index:
                price = prices[symbol]
                trade_value = abs(shares * price)

                # Commission
                commission_cost = trade_value * self.commission

                # Slippage
                slippage_cost = trade_value * self.slippage

                total_cost += (commission_cost + slippage_cost)

        return -total_cost

    def calculate_daily_stats(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate daily portfolio statistics.

        Args:
            positions: Current positions (symbol, shares)
            prices: Price history (date, symbol, price)

        Returns:
            DataFrame with daily portfolio metrics
        """
        # Prepare positions
        if isinstance(positions, pd.DataFrame):
            position_dict = dict(zip(positions['symbol'], positions['shares']))
        else:
            position_dict = positions

        # Prepare prices
        if isinstance(prices.index, pd.MultiIndex):
            prices_wide = prices.unstack(level='symbol')
        else:
            prices_wide = prices

        # Calculate daily values
        daily_stats = []

        for date in prices_wide.index:
            day_prices = prices_wide.loc[date]

            portfolio_value = 0.0
            for symbol, shares in position_dict.items():
                if symbol in day_prices.index:
                    portfolio_value += shares * day_prices[symbol]

            daily_stats.append({
                'date': date,
                'portfolio_value': portfolio_value
            })

        stats_df = pd.DataFrame(daily_stats).set_index('date')
        stats_df['daily_return'] = stats_df['portfolio_value'].pct_change()
        stats_df['cumulative_return'] = (
            stats_df['portfolio_value'] / stats_df['portfolio_value'].iloc[0] - 1
        )

        return stats_df

