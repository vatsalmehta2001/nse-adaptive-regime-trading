"""
Performance Analysis for Backtests.

Comprehensive performance metrics calculation including returns, risk, and
risk-adjusted metrics with regime-specific breakdown.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceAnalyzer:
    """
    Calculate performance metrics for backtests.
    
    Metrics:
    - Returns (total, annualized, CAGR)
    - Risk (volatility, downside deviation)
    - Risk-adjusted (Sharpe, Sortino, Calmar)
    - Drawdowns (max, average, duration)
    - Win rate, profit factor
    - Regime-specific breakdown
    """
    
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (5% default)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/self.TRADING_DAYS_PER_YEAR) - 1
        
        logger.info(f"Initialized performance analyzer with rf={risk_free_rate:.2%}")
    
    def calculate_returns_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate return metrics.
        
        Args:
            returns: Daily returns
            
        Returns:
            Dictionary with:
            - total_return: Cumulative return
            - annualized_return: Annualized return
            - cagr: Compound annual growth rate
            - best_day: Maximum daily return
            - worst_day: Minimum daily return
            - positive_days_pct: Percentage of positive days
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        n_days = len(returns)
        n_years = n_days / self.TRADING_DAYS_PER_YEAR
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # CAGR (same as annualized for this calculation)
        cagr = annualized_return
        
        # Best and worst days
        best_day = float(returns.max())
        worst_day = float(returns.min())
        
        # Positive days
        positive_days_pct = (returns > 0).sum() / len(returns)
        
        metrics = {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'cagr': float(cagr),
            'best_day': best_day,
            'worst_day': worst_day,
            'positive_days_pct': float(positive_days_pct)
        }
        
        return metrics
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            returns: Daily returns
            
        Returns:
            Dictionary with:
            - volatility: Annualized volatility
            - downside_deviation: Downside risk
            - var_95: Value at Risk (95%)
            - cvar_95: Conditional VaR
            - max_drawdown: Maximum drawdown
            - avg_drawdown: Average drawdown
            - max_drawdown_duration: Longest drawdown (days)
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Volatility (annualized)
        volatility = float(returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_deviation = float(
            downside_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        ) if len(downside_returns) > 0 else 0.0
        
        # Value at Risk (95% confidence)
        var_95 = float(returns.quantile(0.05))
        
        # Conditional VaR (expected shortfall)
        cvar_95 = float(returns[returns <= var_95].mean()) if var_95 != 0 else 0.0
        
        # Drawdown metrics
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        
        max_drawdown = float(drawdowns.min())
        avg_drawdown = float(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0.0
        
        # Max drawdown duration
        dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        
        metrics = {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': dd_duration
        }
        
        return metrics
    
    def calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted metrics.
        
        Args:
            returns: Daily returns
            
        Returns:
            Dictionary with:
            - sharpe_ratio: (return - rf) / volatility
            - sortino_ratio: (return - rf) / downside_deviation
            - calmar_ratio: return / max_drawdown
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate needed components
        excess_returns = returns - self.daily_rf
        
        # Annualized metrics
        annual_return = (1 + returns).prod() ** (self.TRADING_DAYS_PER_YEAR / len(returns)) - 1
        annual_excess = annual_return - self.risk_free_rate
        
        volatility = returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio
        sharpe_ratio = float(annual_excess / volatility) if volatility > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < self.daily_rf]
        downside_deviation = (
            downside_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        ) if len(downside_returns) > 0 else 0.0
        
        sortino_ratio = float(annual_excess / downside_deviation) if downside_deviation > 0 else 0.0
        
        # Calmar ratio
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        calmar_ratio = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    
    def calculate_regime_performance(
        self,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Performance breakdown by regime.
        
        Args:
            returns: Daily returns
            regime_labels: Regime label per date
            
        Returns:
            DataFrame with per-regime:
            - total_return
            - sharpe_ratio
            - max_drawdown
            - win_rate
            - days_in_regime
        """
        # Align returns and regime labels
        aligned = pd.DataFrame({
            'returns': returns,
            'regime': regime_labels
        }).dropna()
        
        regime_stats = []
        
        for regime in aligned['regime'].unique():
            regime_returns = aligned[aligned['regime'] == regime]['returns']
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate metrics for this regime
            total_return = (1 + regime_returns).prod() - 1
            
            # Annualized metrics
            volatility = regime_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            annual_return = (
                (1 + regime_returns).prod() ** (self.TRADING_DAYS_PER_YEAR / len(regime_returns)) - 1
            )
            annual_excess = annual_return - self.risk_free_rate
            sharpe_ratio = annual_excess / volatility if volatility > 0 else 0.0
            
            # Drawdown
            equity_curve = (1 + regime_returns).cumprod()
            running_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Win rate
            win_rate = (regime_returns > 0).sum() / len(regime_returns)
            
            regime_stats.append({
                'regime': regime,
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'days_in_regime': len(regime_returns)
            })
        
        return pd.DataFrame(regime_stats)
    
    def calculate_drawdowns(
        self,
        equity_curve: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate all drawdown periods.
        
        Args:
            equity_curve: Portfolio value over time
            
        Returns:
            DataFrame with [start_date, end_date, recovery_date,
                           depth, duration_days]
        """
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        
        drawdown_periods = []
        start_date = None
        max_dd = 0
        
        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start_date is None:
                # Start of drawdown
                start_date = date
                max_dd = drawdowns[date]
            elif is_dd and start_date is not None:
                # Continuing drawdown
                max_dd = min(max_dd, drawdowns[date])
            elif not is_dd and start_date is not None:
                # End of drawdown
                end_idx = i - 1
                end_date = equity_curve.index[end_idx]
                
                # Find recovery date (next time we hit previous high)
                recovery_date = date
                
                duration = (end_date - start_date).days
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'recovery_date': recovery_date,
                    'depth': max_dd,
                    'duration_days': duration
                })
                
                start_date = None
                max_dd = 0
        
        # Handle ongoing drawdown
        if start_date is not None:
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': equity_curve.index[-1],
                'recovery_date': None,
                'depth': max_dd,
                'duration_days': (equity_curve.index[-1] - start_date).days
            })
        
        return pd.DataFrame(drawdown_periods)
    
    def _calculate_max_drawdown_duration(
        self,
        drawdowns: pd.Series
    ) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdowns < 0
        
        if not in_drawdown.any():
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def generate_report(
        self,
        returns: pd.Series,
        regime_labels: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            returns: Daily returns
            regime_labels: Optional regime labels
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Generating performance report...")
        
        report = {}
        
        # Return metrics
        report['returns'] = self.calculate_returns_metrics(returns)
        
        # Risk metrics
        report['risk'] = self.calculate_risk_metrics(returns)
        
        # Risk-adjusted metrics
        report['risk_adjusted'] = self.calculate_risk_adjusted_metrics(returns)
        
        # Regime performance (if available)
        if regime_labels is not None:
            report['regime_performance'] = self.calculate_regime_performance(
                returns, regime_labels
            ).to_dict(orient='records')
        
        # Drawdown analysis
        equity_curve = (1 + returns).cumprod()
        drawdowns_df = self.calculate_drawdowns(equity_curve)
        if len(drawdowns_df) > 0:
            report['top_drawdowns'] = drawdowns_df.nlargest(
                5, 'depth', keep='all'
            ).to_dict(orient='records')
        
        logger.info("Performance report generated")
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Print human-readable summary of performance report.
        
        Args:
            report: Report dictionary from generate_report()
        """
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Returns
        print("\nRETURNS:")
        returns = report.get('returns', {})
        print(f"  Total Return:       {returns.get('total_return', 0):.2%}")
        print(f"  Annualized Return:  {returns.get('annualized_return', 0):.2%}")
        print(f"  CAGR:               {returns.get('cagr', 0):.2%}")
        print(f"  Best Day:           {returns.get('best_day', 0):.2%}")
        print(f"  Worst Day:          {returns.get('worst_day', 0):.2%}")
        print(f"  Positive Days:      {returns.get('positive_days_pct', 0):.1%}")
        
        # Risk
        print("\nRISK:")
        risk = report.get('risk', {})
        print(f"  Volatility:         {risk.get('volatility', 0):.2%}")
        print(f"  Downside Deviation: {risk.get('downside_deviation', 0):.2%}")
        print(f"  Max Drawdown:       {risk.get('max_drawdown', 0):.2%}")
        print(f"  Avg Drawdown:       {risk.get('avg_drawdown', 0):.2%}")
        print(f"  Max DD Duration:    {risk.get('max_drawdown_duration', 0):.0f} days")
        print(f"  VaR (95%):          {risk.get('var_95', 0):.2%}")
        
        # Risk-adjusted
        print("\nRISK-ADJUSTED:")
        risk_adj = report.get('risk_adjusted', {})
        print(f"  Sharpe Ratio:       {risk_adj.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:      {risk_adj.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar Ratio:       {risk_adj.get('calmar_ratio', 0):.3f}")
        
        # Regime performance
        if 'regime_performance' in report:
            print("\nREGIME PERFORMANCE:")
            for regime in report['regime_performance']:
                print(f"  Regime {regime.get('regime')}:")
                print(f"    Return:     {regime.get('total_return', 0):.2%}")
                print(f"    Sharpe:     {regime.get('sharpe_ratio', 0):.3f}")
                print(f"    Max DD:     {regime.get('max_drawdown', 0):.2%}")
                print(f"    Win Rate:   {regime.get('win_rate', 0):.1%}")
                print(f"    Days:       {regime.get('days_in_regime', 0)}")
        
        print("\n" + "="*80 + "\n")

