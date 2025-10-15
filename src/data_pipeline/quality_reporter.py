"""
Data Quality Reporting Module.

Generates comprehensive reports on data quality issues with audit trails.
Provides quality scores and actionable insights for data validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger


class DataQualityReporter:
    """
    Generate data quality reports.
    
    Features:
    - Comprehensive quality metrics
    - Quality scoring (0-100)
    - Audit trail tracking
    - Multi-format export (JSON, CSV, HTML)
    - Cross-symbol comparison
    """
    
    def __init__(self, output_dir: str = "reports/data_quality"):
        """
        Initialize quality reporter.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataQualityReporter initialized: {self.output_dir}")
    
    def generate_quality_report(
        self,
        df: pd.DataFrame,
        audit_trail: Dict,
        symbol: str = "ALL"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            df: Cleaned DataFrame
            audit_trail: Audit information from cleaning
            symbol: Symbol identifier
            
        Returns:
            Quality metrics dictionary
        """
        logger.info(f"Generating quality report for {symbol}...")
        
        report = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None,
                'total_days': (df['date'].max() - df['date'].min()).days if 'date' in df.columns else 0
            },
            'data_quality': {
                'total_rows': len(df),
                'retention_rate': audit_trail.get('retention_rate', 1.0),
                'filtered_rows': audit_trail.get('total_filtered', 0),
                'threshold_used': audit_trail.get('threshold_used', 0.20)
            },
            'missing_data': self._check_missing_data(df),
            'extreme_values': audit_trail.get('filtered_by_symbol', {}),
            'data_issues': self._identify_issues(df)
        }
        
        # Calculate quality score (0-100)
        report['quality_score'] = self._calculate_quality_score(report)
        report['quality_grade'] = self._get_quality_grade(report['quality_score'])
        
        # Save report
        self._save_report(report, symbol)
        
        logger.info(f"Quality Score: {report['quality_score']:.1f}/100 ({report['quality_grade']})")
        
        return report
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing data points.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Missing data metrics
        """
        if 'date' not in df.columns:
            return {'completeness': 1.0}
        
        expected_days = (df['date'].max() - df['date'].min()).days
        actual_days = len(df['date'].unique())
        
        return {
            'expected_days': expected_days,
            'actual_days': actual_days,
            'missing_days': max(0, expected_days - actual_days),
            'completeness': actual_days / expected_days if expected_days > 0 else 0.0
        }
    
    def _identify_issues(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify specific data quality issues.
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for zero volume days
        if 'volume' in df.columns:
            zero_vol = (df['volume'] == 0).sum()
            if zero_vol > 0:
                issues.append({
                    'type': 'zero_volume',
                    'count': int(zero_vol),
                    'severity': 'low',
                    'description': f"{zero_vol} days with zero volume"
                })
        
        # Check for constant prices
        if 'close' in df.columns and 'symbol' in df.columns:
            df_temp = df.copy()
            df_temp['price_change'] = df_temp.groupby('symbol')['close'].diff()
            constant_price = (df_temp['price_change'] == 0).sum()
            
            if constant_price > 5:
                issues.append({
                    'type': 'constant_price',
                    'count': int(constant_price),
                    'severity': 'medium',
                    'description': f"{constant_price} days with no price change"
                })
        
        # Check for suspicious patterns (volume drops)
        if 'volume' in df.columns and len(df) > 40:
            recent_vol = df.tail(20)['volume'].mean()
            older_vol = df.head(20)['volume'].mean()
            
            if older_vol > 0 and recent_vol < older_vol * 0.1:
                issues.append({
                    'type': 'volume_drop',
                    'severity': 'high',
                    'description': f"Recent volume is {recent_vol/older_vol*100:.1f}% of historical",
                    'details': {
                        'recent_volume': float(recent_vol),
                        'historical_volume': float(older_vol)
                    }
                })
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            issues.append({
                'type': 'missing_values',
                'count': int(nan_counts.sum()),
                'severity': 'medium',
                'description': f"{nan_counts.sum()} missing values across columns",
                'details': nan_counts[nan_counts > 0].to_dict()
            })
        
        return issues
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """
        Calculate overall quality score (0-100).
        
        Args:
            report: Quality report dictionary
            
        Returns:
            Quality score (0-100)
        """
        score = 100.0
        
        # Deduct for low retention (up to -20 points)
        retention = report['data_quality']['retention_rate']
        score -= (1 - retention) * 20
        
        # Deduct for missing data (up to -30 points)
        completeness = report['missing_data'].get('completeness', 1.0)
        score -= (1 - completeness) * 30
        
        # Deduct for data issues
        for issue in report['data_issues']:
            severity_deductions = {
                'low': 5,
                'medium': 10,
                'high': 20
            }
            score -= severity_deductions.get(issue.get('severity', 'low'), 0)
        
        return max(0.0, min(100.0, score))
    
    def _get_quality_grade(self, score: float) -> str:
        """
        Convert quality score to grade.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Quality grade
        """
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 60:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _save_report(self, report: Dict, symbol: str):
        """
        Save report to JSON and CSV.
        
        Args:
            report: Quality report
            symbol: Symbol identifier
        """
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report (full details)
        json_path = self.output_dir / f"quality_report_{symbol}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # CSV summary
        summary = {
            'symbol': symbol,
            'timestamp': report['timestamp'],
            'quality_score': report['quality_score'],
            'quality_grade': report['quality_grade'],
            'retention_rate': report['data_quality']['retention_rate'],
            'completeness': report['missing_data'].get('completeness', 0),
            'issues_count': len(report['data_issues']),
            'total_rows': report['data_quality']['total_rows']
        }
        
        csv_path = self.output_dir / f"quality_summary_{symbol}_{timestamp}.csv"
        pd.DataFrame([summary]).to_csv(csv_path, index=False)
        
        logger.info(f"Quality report saved: {json_path}")
    
    def generate_comparison_report(self, reports: List[Dict]) -> pd.DataFrame:
        """
        Compare quality across multiple symbols.
        
        Args:
            reports: List of quality reports
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Generating comparison report for {len(reports)} symbols...")
        
        comparison = pd.DataFrame([
            {
                'symbol': r['symbol'],
                'quality_score': r['quality_score'],
                'quality_grade': r['quality_grade'],
                'retention_rate': r['data_quality']['retention_rate'],
                'completeness': r['missing_data'].get('completeness', 0),
                'issues': len(r['data_issues']),
                'filtered_rows': r['data_quality']['filtered_rows']
            }
            for r in reports
        ])
        
        comparison = comparison.sort_values('quality_score', ascending=False)
        
        # Calculate summary statistics
        summary_stats = {
            'total_symbols': len(comparison),
            'avg_quality_score': comparison['quality_score'].mean(),
            'excellent_count': (comparison['quality_score'] >= 90).sum(),
            'good_count': ((comparison['quality_score'] >= 75) & (comparison['quality_score'] < 90)).sum(),
            'acceptable_count': ((comparison['quality_score'] >= 60) & (comparison['quality_score'] < 75)).sum(),
            'poor_count': (comparison['quality_score'] < 60).sum()
        }
        
        logger.info(f"Quality Summary: {summary_stats['excellent_count']} excellent, "
                   f"{summary_stats['good_count']} good, {summary_stats['acceptable_count']} acceptable, "
                   f"{summary_stats['poor_count']} poor")
        
        # Save
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = self.output_dir / f"comparison_report_{timestamp}.csv"
        comparison.to_csv(comparison_path, index=False)
        
        stats_path = self.output_dir / f"summary_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=int)
        
        logger.info(f"Comparison report saved: {comparison_path}")
        
        return comparison
    
    def print_quality_summary(self, report: Dict):
        """
        Print human-readable quality summary.
        
        Args:
            report: Quality report dictionary
        """
        print("\n" + "="*80)
        print(f"DATA QUALITY REPORT: {report['symbol']}")
        print("="*80)
        
        print(f"\nQuality Score: {report['quality_score']:.1f}/100 ({report['quality_grade']})")
        
        print("\nDATA STATISTICS:")
        print(f"  Total Rows:      {report['data_quality']['total_rows']:,}")
        print(f"  Filtered Rows:   {report['data_quality']['filtered_rows']:,}")
        print(f"  Retention Rate:  {report['data_quality']['retention_rate']*100:.1f}%")
        print(f"  Completeness:    {report['missing_data'].get('completeness', 0)*100:.1f}%")
        
        if report['data_range']['start']:
            print(f"\nDATE RANGE:")
            print(f"  Start: {report['data_range']['start']}")
            print(f"  End:   {report['data_range']['end']}")
            print(f"  Days:  {report['data_range']['total_days']}")
        
        if report['extreme_values']:
            print(f"\nEXTREME VALUES FILTERED:")
            for symbol, details in report['extreme_values'].items():
                print(f"  {symbol}: {details['count']} dates")
                print(f"    Max: {details['max_return']*100:+.1f}%, Min: {details['min_return']*100:+.1f}%")
        
        if report['data_issues']:
            print(f"\nDATA ISSUES ({len(report['data_issues'])}):")
            for issue in report['data_issues']:
                severity_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}.get(issue['severity'], "•")
                print(f"  {severity_icon} [{issue['severity'].upper()}] {issue['type']}: {issue.get('description', 'N/A')}")
        
        print("\n" + "="*80 + "\n")

