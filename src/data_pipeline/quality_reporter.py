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
        symbol: str = "ALL",
        formats: List[str] = ['json', 'csv']
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            df: Cleaned DataFrame
            audit_trail: Audit information from cleaning
            symbol: Symbol identifier
            formats: List of output formats ('json', 'csv', 'html')
            
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
        
        # Generate HTML if requested
        if 'html' in formats:
            self.generate_html_report(report, symbol)
        
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
    
    def generate_html_report(self, report: Dict, symbol: str) -> str:
        """
        Generate HTML quality report.
        
        Args:
            report: Quality report dictionary
            symbol: Symbol identifier
            
        Returns:
            Path to saved HTML file
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report - {symbol}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .score {{ font-size: 48px; font-weight: bold; text-align: center; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .excellent {{ background: #4CAF50; color: white; }}
        .good {{ background: #8BC34A; color: white; }}
        .acceptable {{ background: #FFC107; color: white; }}
        .poor {{ background: #F44336; color: white; }}
        .metric {{ display: inline-block; width: 23%; margin: 10px 1%; padding: 15px; background: #f9f9f9; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #2196F3; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .issue-high {{ color: #F44336; font-weight: bold; }}
        .issue-medium {{ color: #FF9800; }}
        .issue-low {{ color: #9E9E9E; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Quality Report: {symbol}</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <div class="score {grade_class}">
            {quality_score:.1f} / 100
            <div style="font-size: 20px; margin-top: 10px;">{quality_grade}</div>
        </div>
        
        <h2>Key Metrics</h2>
        <div>
            <div class="metric">
                <div class="metric-value">{total_rows:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{retention_rate:.1f}%</div>
                <div class="metric-label">Retention Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{completeness:.1f}%</div>
                <div class="metric-label">Completeness</div>
            </div>
            <div class="metric">
                <div class="metric-value">{filtered_rows:,}</div>
                <div class="metric-label">Filtered Rows</div>
            </div>
        </div>
        
        <h2>Date Range</h2>
        <p><strong>Start:</strong> {date_start} | <strong>End:</strong> {date_end} | <strong>Days:</strong> {total_days}</p>
        
        {extreme_values_section}
        {issues_section}
    </div>
</body>
</html>
        """
        
        # Format grade class
        grade_class = report['quality_grade'].lower()
        
        # Format extreme values section
        extreme_values_section = ""
        if report['extreme_values']:
            extreme_values_section = "<h2>Extreme Values Filtered</h2><table><tr><th>Symbol</th><th>Count</th><th>Max Return</th><th>Min Return</th></tr>"
            for sym, details in report['extreme_values'].items():
                extreme_values_section += f"<tr><td>{sym}</td><td>{details['count']}</td><td>{details['max_return']*100:+.2f}%</td><td>{details['min_return']*100:+.2f}%</td></tr>"
            extreme_values_section += "</table>"
        
        # Format issues section
        issues_section = ""
        if report['data_issues']:
            issues_section = "<h2>Data Issues</h2><table><tr><th>Severity</th><th>Type</th><th>Description</th></tr>"
            for issue in report['data_issues']:
                severity_class = f"issue-{issue['severity']}"
                issues_section += f"<tr class='{severity_class}'><td>{issue['severity'].upper()}</td><td>{issue['type']}</td><td>{issue.get('description', 'N/A')}</td></tr>"
            issues_section += "</table>"
        
        # Fill template
        html_content = html_template.format(
            symbol=report['symbol'],
            timestamp=report['timestamp'],
            quality_score=report['quality_score'],
            quality_grade=report['quality_grade'],
            grade_class=grade_class,
            total_rows=report['data_quality']['total_rows'],
            retention_rate=report['data_quality']['retention_rate']*100,
            completeness=report['missing_data'].get('completeness', 1.0)*100,
            filtered_rows=report['data_quality']['filtered_rows'],
            date_start=report['data_range'].get('start', 'N/A'),
            date_end=report['data_range'].get('end', 'N/A'),
            total_days=report['data_range'].get('total_days', 0),
            extreme_values_section=extreme_values_section,
            issues_section=issues_section
        )
        
        # Save HTML file
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        html_path = self.output_dir / f"quality_report_{symbol}_{timestamp}.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved: {html_path}")
        return str(html_path)

