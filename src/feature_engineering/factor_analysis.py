"""
Factor Analysis and Validation Toolkit.

Provides tools for analyzing alpha factors:
- IC (Information Coefficient) calculation
- Correlation analysis
- VIF (Variance Inflation Factor)
- Regime-specific analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr


class FactorAnalyzer:
    """
    Factor validation and analysis toolkit.

    Capabilities:
    - IC (Information Coefficient) calculation
    - Correlation analysis
    - VIF (Variance Inflation Factor) for multicollinearity
    - Regime-specific IC
    - Factor decay analysis
    """

    def __init__(self):
        """Initialize factor analyzer."""
        pass

    def calculate_ic(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        periods: List[int] = [1, 5, 20],
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Calculate IC for each factor at different horizons.

        IC = correlation(factor_value_t, return_t+n)

        Args:
            factors: DataFrame with factors
            forward_returns: Series with returns (already shifted forward)
            periods: List of forward periods to calculate
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            DataFrame with columns [factor_name, ic_1d, ic_5d, ic_20d, mean_ic]
        """
        logger.info(f"Calculating IC for {len(factors.columns)} factors...")

        ic_results = []

        # Get correlation function
        corr_func = spearmanr if method == "spearman" else pearsonr

        for factor_name in factors.columns:
            if factor_name.startswith('factor_'):
                ic_values = {'factor': factor_name}

                for period in periods:
                    # Align data
                    factor_values = factors[factor_name]
                    returns = forward_returns.shift(-period)

                    # Remove NaN
                    valid_mask = factor_values.notna() & returns.notna()

                    if valid_mask.sum() > 10:
                        # Calculate correlation
                        corr, pval = corr_func(
                            factor_values[valid_mask],
                            returns[valid_mask]
                        )

                        ic_values[f'ic_{period}d'] = corr
                        ic_values[f'pval_{period}d'] = pval
                    else:
                        ic_values[f'ic_{period}d'] = np.nan
                        ic_values[f'pval_{period}d'] = np.nan

                # Calculate mean IC
                ic_cols = [f'ic_{p}d' for p in periods]
                ic_values['mean_ic'] = np.nanmean([ic_values.get(col, np.nan) for col in ic_cols])
                ic_values['abs_mean_ic'] = abs(ic_values['mean_ic'])

                ic_results.append(ic_values)

        ic_df = pd.DataFrame(ic_results)
        ic_df = ic_df.sort_values('abs_mean_ic', ascending=False)

        logger.info(f" IC calculated. Mean IC: {ic_df['mean_ic'].mean():.4f}")

        return ic_df

    def calculate_ic_by_regime(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        regime_labels: pd.Series,
        period: int = 5
    ) -> pd.DataFrame:
        """
        Calculate IC separately for each regime.

        Args:
            factors: Factor DataFrame
            forward_returns: Forward returns
            regime_labels: Regime labels for each date
            period: Forward period for returns

        Returns:
            DataFrame with IC by regime
        """
        logger.info("Calculating regime-specific IC...")

        results = []

        # Get unique regimes
        unique_regimes = regime_labels.unique()

        for factor_name in factors.columns:
            if factor_name.startswith('factor_'):
                factor_ic = {'factor': factor_name}

                for regime in unique_regimes:
                    # Filter by regime
                    regime_mask = regime_labels == regime

                    factor_values = factors[factor_name][regime_mask]
                    returns = forward_returns.shift(-period)[regime_mask]

                    # Remove NaN
                    valid_mask = factor_values.notna() & returns.notna()

                    if valid_mask.sum() > 10:
                        corr, _ = spearmanr(
                            factor_values[valid_mask],
                            returns[valid_mask]
                        )
                        factor_ic[f'ic_regime_{regime}'] = corr
                    else:
                        factor_ic[f'ic_regime_{regime}'] = np.nan

                results.append(factor_ic)

        return pd.DataFrame(results)

    def analyze_correlation(
        self,
        factors: pd.DataFrame,
        threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Find highly correlated factor pairs (|corr| > threshold).

        Args:
            factors: Factor DataFrame
            threshold: Correlation threshold

        Returns:
            DataFrame with correlated pairs
        """
        logger.info(f"Analyzing factor correlations (threshold={threshold})...")

        # Calculate correlation matrix
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        corr_matrix = factors[factor_cols].corr()

        # Find pairs above threshold
        high_corr_pairs = []

        for i in range(len(factor_cols)):
            for j in range(i+1, len(factor_cols)):
                corr = corr_matrix.iloc[i, j]

                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'factor_1': factor_cols[i],
                        'factor_2': factor_cols[j],
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })

        corr_df = pd.DataFrame(high_corr_pairs)

        if not corr_df.empty:
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            logger.info(f"Found {len(corr_df)} highly correlated pairs")
        else:
            logger.info("No highly correlated pairs found")

        return corr_df

    def calculate_vif(
        self,
        factors: pd.DataFrame,
        sample_size: int = 1000
    ) -> pd.Series:
        """
        Variance Inflation Factor for each feature.

        VIF = 1 / (1 - R²)
        where R² is from regressing factor on all others.

        Interpretation:
        - VIF < 5: Low multicollinearity
        - VIF 5-10: Moderate
        - VIF > 10: High (consider removing)

        Args:
            factors: Factor DataFrame
            sample_size: Number of samples to use (for speed)

        Returns:
            Series with VIF for each factor
        """
        logger.info("Calculating VIF for multicollinearity...")

        # Get factor columns
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]

        # Sample if needed
        if len(factors) > sample_size:
            factors_sample = factors[factor_cols].sample(sample_size, random_state=42)
        else:
            factors_sample = factors[factor_cols].copy()

        # Remove NaN
        factors_sample = factors_sample.dropna()

        vif_data = {}

        for i, col in enumerate(factor_cols[:50]):  # Limit to first 50 for speed
            # Regress this factor on all others
            X = factors_sample.drop(columns=[col])
            y = factors_sample[col]

            try:
                # Fit regression
                lr = LinearRegression()
                lr.fit(X, y)

                # Calculate R²
                r_squared = lr.score(X, y)

                # Calculate VIF
                vif = 1 / (1 - r_squared) if r_squared < 0.999 else 999

                vif_data[col] = vif

            except Exception as e:
                logger.warning(f"Could not calculate VIF for {col}: {e}")
                vif_data[col] = np.nan

        vif_series = pd.Series(vif_data).sort_values(ascending=False)

        logger.info(f" VIF calculated. Median VIF: {vif_series.median():.2f}")

        return vif_series

    def select_top_factors(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        n: int = 50,
        remove_correlated: bool = True,
        corr_threshold: float = 0.8
    ) -> List[str]:
        """
        Select top N factors by IC, removing highly correlated pairs.

        Algorithm:
        1. Calculate IC for all factors
        2. Sort by absolute IC
        3. Iteratively select factors, skipping those correlated >0.8 with already-selected

        Args:
            factors: Factor DataFrame
            forward_returns: Forward returns
            n: Number of factors to select
            remove_correlated: Whether to remove correlated factors
            corr_threshold: Correlation threshold

        Returns:
            List of selected factor names
        """
        logger.info(f"Selecting top {n} factors...")

        # Calculate IC
        ic_df = self.calculate_ic(factors, forward_returns, periods=[5])
        ic_df = ic_df.sort_values('abs_mean_ic', ascending=False)

        if not remove_correlated:
            return ic_df['factor'].head(n).tolist()

        # Calculate correlation matrix
        factor_cols = ic_df['factor'].tolist()
        corr_matrix = factors[factor_cols].corr()

        # Iteratively select factors
        selected = []

        for factor in factor_cols:
            if len(selected) >= n:
                break

            # Check correlation with already selected factors
            is_correlated = False

            for selected_factor in selected:
                if abs(corr_matrix.loc[factor, selected_factor]) > corr_threshold:
                    is_correlated = True
                    break

            if not is_correlated:
                selected.append(factor)

        logger.info(f" Selected {len(selected)} factors (removed correlated)")

        return selected

    def generate_report(
        self,
        factors: pd.DataFrame,
        returns: pd.Series,
        regime_labels: pd.Series = None,
        output_path: str = "reports/factor_analysis.html"
    ):
        """
        Generate comprehensive HTML report with visualizations.

        Args:
            factors: Factor DataFrame
            returns: Returns series
            regime_labels: Optional regime labels
            output_path: Path to save HTML report
        """
        logger.info("Generating factor analysis report...")

        # Calculate metrics
        ic_df = self.calculate_ic(factors, returns)
        corr_pairs = self.analyze_correlation(factors)

        # Build HTML report
        html = f"""
        <html>
        <head>
            <title>Factor Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Factor Analysis Report</h1>

            <div class="metric">
                <h2>Summary Statistics</h2>
                <p>Total Factors: {len(factors.columns)}</p>
                <p>Mean IC: {ic_df['mean_ic'].mean():.4f}</p>
                <p>Median IC: {ic_df['mean_ic'].median():.4f}</p>
                <p>Highly Correlated Pairs: {len(corr_pairs)}</p>
            </div>

            <h2>Top 20 Factors by IC</h2>
            {ic_df.head(20).to_html()}

            <h2>Highly Correlated Factor Pairs</h2>
            {corr_pairs.head(20).to_html() if not corr_pairs.empty else '<p>No highly correlated pairs found</p>'}
        </body>
        </html>
        """

        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f" Report saved to {output_path}")

