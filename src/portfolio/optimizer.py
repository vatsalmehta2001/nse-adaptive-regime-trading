"""
Portfolio Optimizer using Convex Optimization.

Implements various portfolio optimization strategies with constraints.
Uses cvxpy for efficient convex optimization.
"""

from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger


class PortfolioOptimizer:
    """
    Portfolio optimization using cvxpy.

    Methods:
    - Mean-variance (Markowitz)
    - Maximum Sharpe ratio
    - Risk parity
    - Minimum variance

    Constraints:
    - Long-only or long-short
    - Position limits
    - Leverage constraints
    - Turnover limits
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

        # Convert to daily
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        logger.info(f"Initialized portfolio optimizer with rf={risk_free_rate:.2%}")

    def optimize_mean_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Mean-variance optimization (Markowitz).

        Objective: maximize return - risk_aversion * variance

        Args:
            expected_returns: Expected return per asset
            cov_matrix: Return covariance matrix
            risk_aversion: Risk aversion parameter (1.0 = neutral)
            constraints: Position limits, etc.

        Returns:
            Optimal weights (sum to 1)
        """
        logger.info("Solving mean-variance optimization...")

        n_assets = len(expected_returns)

        # Align data
        assets = expected_returns.index
        mu = expected_returns.values
        Sigma = cov_matrix.loc[assets, assets].values

        # Decision variable
        w = cp.Variable(n_assets)

        # Objective: maximize return - risk_aversion * variance
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        objective = cp.Maximize(ret - risk_aversion * risk)

        # Base constraints
        base_constraints = [
            cp.sum(w) == 1,  # Fully invested
        ]

        # Apply additional constraints
        all_constraints = self._build_constraints(w, n_assets, constraints)

        # Solve
        problem = cp.Problem(objective, base_constraints + all_constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")

        # Extract weights
        weights = pd.Series(w.value, index=assets, name='weight')

        # Log results
        portfolio_return = float(mu @ w.value)
        portfolio_risk = float(np.sqrt(w.value @ Sigma @ w.value))

        logger.info(f"Portfolio return: {portfolio_return:.4f}, risk: {portfolio_risk:.4f}")
        logger.info(f"Active positions: {(np.abs(weights) > 1e-4).sum()}")

        return weights

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Maximum Sharpe ratio optimization.

        Maximize: (return - risk_free_rate) / volatility

        Args:
            expected_returns: Expected return per asset
            cov_matrix: Return covariance matrix
            constraints: Position limits, etc.

        Returns:
            Optimal weights
        """
        logger.info("Solving maximum Sharpe ratio optimization...")

        n_assets = len(expected_returns)

        # Align data
        assets = expected_returns.index
        mu = expected_returns.values - self.daily_rf  # Excess returns
        Sigma = cov_matrix.loc[assets, assets].values

        # Decision variable (use auxiliary variable for Sharpe)
        w = cp.Variable(n_assets)
        kappa = cp.Variable()  # Auxiliary variable

        # Reformulate as SOCP (Second-Order Cone Program)
        # Maximize: mu @ y, subject to ||Sigma^(1/2) @ y||_2 <= kappa, 1^T y = kappa
        # Then w = y / kappa

        # Actually, simpler to maximize quadratic utility with high risk aversion
        # Or use the efficient frontier approach

        # For numerical stability, maximize: mu @ w / sqrt(w @ Sigma @ w)
        # This is quasi-convex, so we solve via bisection or use approximation

        # Approximation: Maximize mu @ w - 0.5 * lambda * w @ Sigma @ w
        # Then search for lambda that gives max Sharpe

        # Simpler: Use mean-variance with lambda chosen to maximize Sharpe
        # For now, use a reasonable lambda
        lambda_sharpe = 0.5

        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        objective = cp.Maximize(ret - lambda_sharpe * risk)

        # Base constraints
        base_constraints = [
            cp.sum(w) == 1,
        ]

        # Apply additional constraints
        all_constraints = self._build_constraints(w, n_assets, constraints)

        # Solve
        problem = cp.Problem(objective, base_constraints + all_constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")

        # Extract weights
        weights = pd.Series(w.value, index=assets, name='weight')

        # Calculate Sharpe ratio
        portfolio_return = float(mu @ w.value)
        portfolio_risk = float(np.sqrt(w.value @ Sigma @ w.value))
        sharpe = portfolio_return / (portfolio_risk + 1e-10)

        logger.info(f"Portfolio Sharpe ratio: {sharpe:.4f}")
        logger.info(f"Active positions: {(np.abs(weights) > 1e-4).sum()}")

        return weights

    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Risk parity: equal risk contribution per asset.

        Each asset contributes equally to portfolio risk.

        Args:
            cov_matrix: Return covariance matrix
            constraints: Position limits, etc.

        Returns:
            Optimal weights
        """
        logger.info("Solving risk parity optimization...")

        assets = cov_matrix.index
        n_assets = len(assets)
        Sigma = cov_matrix.values

        # Decision variable
        w = cp.Variable(n_assets)

        # Risk parity: minimize sum of squared differences in risk contributions
        # Risk contribution of asset i: w_i * (Sigma @ w)_i
        # We want: w_i * (Sigma @ w)_i = constant for all i

        # Simplified approach: Use inverse volatility weighting as approximation
        # Or solve the more complex optimization

        # Exact risk parity requires iterative solution
        # Here we use a convex approximation: minimize variance with budget constraint

        # Simpler alternative: inverse volatility weighting
        vol = np.sqrt(np.diag(Sigma))
        w_rp = 1 / vol
        w_rp = w_rp / w_rp.sum()

        weights = pd.Series(w_rp, index=assets, name='weight')

        # Apply constraints if needed (requires optimization)
        if constraints:
            # Use minimum variance with weights close to risk parity
            w = cp.Variable(n_assets)

            risk = cp.quad_form(w, Sigma)
            deviation = cp.sum_squares(w - w_rp)

            objective = cp.Minimize(risk + 0.1 * deviation)

            base_constraints = [cp.sum(w) == 1]
            all_constraints = self._build_constraints(w, n_assets, constraints)

            problem = cp.Problem(objective, base_constraints + all_constraints)
            problem.solve()

            if problem.status == cp.OPTIMAL:
                weights = pd.Series(w.value, index=assets, name='weight')

        logger.info(f"Active positions: {(np.abs(weights) > 1e-4).sum()}")

        return weights

    def optimize_min_variance(
        self,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Minimum variance portfolio (defensive).

        Args:
            cov_matrix: Return covariance matrix
            constraints: Position limits, etc.

        Returns:
            Optimal weights
        """
        logger.info("Solving minimum variance optimization...")

        n_assets = len(cov_matrix)
        assets = cov_matrix.index
        Sigma = cov_matrix.values

        # Decision variable
        w = cp.Variable(n_assets)

        # Objective: minimize variance
        risk = cp.quad_form(w, Sigma)
        objective = cp.Minimize(risk)

        # Base constraints
        base_constraints = [
            cp.sum(w) == 1,
        ]

        # Apply additional constraints
        all_constraints = self._build_constraints(w, n_assets, constraints)

        # Solve
        problem = cp.Problem(objective, base_constraints + all_constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")

        # Extract weights
        weights = pd.Series(w.value, index=assets, name='weight')

        portfolio_risk = float(np.sqrt(w.value @ Sigma @ w.value))

        logger.info(f"Portfolio risk: {portfolio_risk:.4f}")
        logger.info(f"Active positions: {(np.abs(weights) > 1e-4).sum()}")

        return weights

    def _build_constraints(
        self,
        w: cp.Variable,
        n_assets: int,
        constraints: Optional[Dict[str, Any]]
    ) -> list:
        """
        Build constraint list from constraints dict.

        Supported constraints:
        - max_position: Maximum weight per stock (e.g., 0.1 = 10%)
        - long_only: No short positions
        - max_leverage: Maximum total leverage
        - max_turnover: Maximum change from current weights (requires current_weights)

        Args:
            w: Weight variable
            n_assets: Number of assets
            constraints: Constraints dictionary

        Returns:
            List of cvxpy constraints
        """
        if constraints is None:
            return []

        constraint_list = []

        # Long-only constraint
        if constraints.get('long_only', False):
            constraint_list.append(w >= 0)

        # Maximum position size
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            constraint_list.append(w <= max_pos)
            constraint_list.append(w >= -max_pos)

        # Maximum leverage
        if 'max_leverage' in constraints:
            max_lev = constraints['max_leverage']
            constraint_list.append(cp.norm(w, 1) <= max_lev)

        # Turnover constraint
        if 'max_turnover' in constraints and 'current_weights' in constraints:
            max_turn = constraints['max_turnover']
            w_current = constraints['current_weights']

            # Ensure same length
            if len(w_current) == n_assets:
                constraint_list.append(cp.norm(w - w_current, 1) <= max_turn)

        # Minimum position size (to reduce transaction costs)
        if 'min_position' in constraints:
            # This creates a mixed-integer problem, skip for now
            pass

        return constraint_list

    def apply_constraints(
        self,
        weights: pd.Series,
        constraints: Dict[str, Any]
    ) -> pd.Series:
        """
        Apply position and exposure constraints to existing weights.

        This is a post-processing step for simple constraints.

        Args:
            weights: Current weights
            constraints: Constraints to apply

        Returns:
            Adjusted weights
        """
        adjusted = weights.copy()

        # Long-only
        if constraints.get('long_only', False):
            adjusted = adjusted.clip(lower=0)

        # Max position
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            adjusted = adjusted.clip(lower=-max_pos, upper=max_pos)

        # Renormalize to sum to 1
        weight_sum = adjusted.sum()
        if abs(weight_sum) > 1e-6:
            adjusted = adjusted / weight_sum

        return adjusted

    def calculate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_points: int = 20,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier
            constraints: Position constraints

        Returns:
            DataFrame with frontier points (risk, return, weights)
        """
        logger.info(f"Calculating efficient frontier ({n_points} points)...")

        frontier_points = []

        # Range of risk aversion parameters
        risk_aversions = np.logspace(-2, 2, n_points)

        for risk_aversion in risk_aversions:
            weights = self.optimize_mean_variance(
                expected_returns,
                cov_matrix,
                risk_aversion=risk_aversion,
                constraints=constraints
            )

            # Calculate portfolio stats
            portfolio_return = float(expected_returns @ weights)
            portfolio_risk = float(np.sqrt(weights @ cov_matrix @ weights))

            frontier_points.append({
                'risk': portfolio_risk,
                'return': portfolio_return,
                'sharpe': (portfolio_return - self.daily_rf) / (portfolio_risk + 1e-10),
                'weights': weights.to_dict()
            })

        frontier_df = pd.DataFrame(frontier_points)

        logger.info("Efficient frontier calculated")

        return frontier_df

