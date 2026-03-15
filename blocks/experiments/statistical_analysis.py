"""
Statistical Analysis Framework for Retrieval Methods Research
==============================================================

This module provides comprehensive statistical analysis tools for comparing
retrieval methods with rigorous statistical testing and visualization.

Features:
- Descriptive statistics
- Parametric tests (t-test, ANOVA)
- Non-parametric tests (Wilcoxon, Friedman, Kruskal-Wallis)
- Effect size calculations (Cohen's d, η²)
- Multiple comparison corrections (Bonferroni, Holm)
- Correlation analysis
- Visualization (box plots, violin plots, heatmaps)
- LaTeX table generation
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, friedmanchisquare, wilcoxon
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


@dataclass
class StatisticalTest:
    """Results of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'interpretation': self.interpretation
        }


class DescriptiveStatistics:
    """
    Calculate descriptive statistics for experimental results.
    """
    
    @staticmethod
    def compute_descriptives(
        data: pd.DataFrame,
        metrics: List[str],
        group_by: str = 'strategy'
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics (mean, std, median, etc.) by group.
        
        Args:
            data: DataFrame with experimental results
            metrics: List of metric column names
            group_by: Column to group by (default: 'strategy')
            
        Returns:
            DataFrame with descriptive statistics
        """
        results = []
        
        for metric in metrics:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][metric].dropna()
                
                if len(group_data) > 0:
                    stats_dict = {
                        group_by: group,
                        'metric': metric,
                        'n': len(group_data),
                        'mean': group_data.mean(),
                        'std': group_data.std(),
                        'median': group_data.median(),
                        'min': group_data.min(),
                        'max': group_data.max(),
                        'q1': group_data.quantile(0.25),
                        'q3': group_data.quantile(0.75),
                        'iqr': group_data.quantile(0.75) - group_data.quantile(0.25),
                        'cv': (group_data.std() / group_data.mean()) if group_data.mean() != 0 else np.nan
                    }
                    results.append(stats_dict)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def create_summary_table(descriptives: pd.DataFrame) -> str:
        """
        Create a formatted summary table (LaTeX compatible).
        
        Args:
            descriptives: DataFrame from compute_descriptives
            
        Returns:
            LaTeX table string
        """
        # Pivot to wide format
        pivot = descriptives.pivot_table(
            index='metric',
            columns='strategy',
            values=['mean', 'std', 'median']
        )
        
        # Format as LaTeX
        latex = pivot.to_latex(
            float_format="%.4f",
            caption="Descriptive Statistics by Retrieval Strategy",
            label="tab:descriptives"
        )
        
        return latex


class InferentialStatistics:
    """
    Perform inferential statistical tests.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize with significance level.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def test_normality(
        self,
        data: pd.Series,
        test: str = 'shapiro'
    ) -> StatisticalTest:
        """
        Test if data follows normal distribution.
        
        Args:
            data: Series of values
            test: 'shapiro' (Shapiro-Wilk) or 'ks' (Kolmogorov-Smirnov)
            
        Returns:
            StatisticalTest object
        """
        data_clean = data.dropna()
        
        if test == 'shapiro':
            statistic, p_value = shapiro(data_clean)
            test_name = "Shapiro-Wilk"
        else:
            statistic, p_value = stats.kstest(data_clean, 'norm')
            test_name = "Kolmogorov-Smirnov"
        
        significant = p_value < self.alpha
        interpretation = (
            "Data is NOT normally distributed" if significant 
            else "Data appears normally distributed"
        )
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )
    
    def test_homogeneity(
        self,
        *groups: pd.Series
    ) -> StatisticalTest:
        """
        Test homogeneity of variances (Levene's test).
        
        Args:
            groups: Multiple Series representing different groups
            
        Returns:
            StatisticalTest object
        """
        clean_groups = [g.dropna() for g in groups]
        statistic, p_value = levene(*clean_groups)
        
        significant = p_value < self.alpha
        interpretation = (
            "Variances are NOT equal across groups" if significant
            else "Variances are equal across groups (homoscedasticity)"
        )
        
        return StatisticalTest(
            test_name="Levene's Test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )
    
    def paired_t_test(
        self,
        group1: pd.Series,
        group2: pd.Series,
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """
        Paired t-test for comparing two related samples.
        
        Args:
            group1: First group (paired with group2)
            group2: Second group (paired with group1)
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            StatisticalTest object with Cohen's d effect size
        """
        # Align and clean
        aligned = pd.DataFrame({'g1': group1, 'g2': group2}).dropna()
        
        statistic, p_value = stats.ttest_rel(
            aligned['g1'],
            aligned['g2'],
            alternative=alternative
        )
        
        # Calculate Cohen's d for paired samples
        diff = aligned['g1'] - aligned['g2']
        cohens_d = diff.mean() / diff.std()
        
        significant = p_value < self.alpha
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        interpretation = (
            f"{'Significant' if significant else 'No significant'} difference "
            f"(Cohen's d = {cohens_d:.3f}, {effect_interp} effect)"
        )
        
        return StatisticalTest(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=cohens_d,
            interpretation=interpretation
        )
    
    def repeated_measures_anova(
        self,
        data: pd.DataFrame,
        dv: str,
        within: str,
        subject: str
    ) -> StatisticalTest:
        """
        Repeated measures ANOVA (Friedman test for non-parametric).
        
        Args:
            data: DataFrame with long-format data
            dv: Dependent variable (metric)
            within: Within-subject factor (e.g., 'strategy')
            subject: Subject identifier (e.g., 'query_id')
            
        Returns:
            StatisticalTest object with η² effect size
        """
        # Pivot to wide format for Friedman test
        pivoted = data.pivot_table(
            values=dv,
            index=subject,
            columns=within
        ).dropna()
        
        # Friedman test (non-parametric repeated measures)
        statistic, p_value = friedmanchisquare(*[pivoted[col] for col in pivoted.columns])
        
        # Calculate effect size (Kendall's W)
        n = len(pivoted)
        k = len(pivoted.columns)
        kendalls_w = statistic / (n * (k - 1))
        
        significant = p_value < self.alpha
        
        # Interpret effect size
        if kendalls_w < 0.1:
            effect_interp = "negligible"
        elif kendalls_w < 0.3:
            effect_interp = "small"
        elif kendalls_w < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        interpretation = (
            f"{'Significant' if significant else 'No significant'} differences among methods "
            f"(Kendall's W = {kendalls_w:.3f}, {effect_interp} effect)"
        )
        
        return StatisticalTest(
            test_name="Friedman Test (Repeated Measures)",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=kendalls_w,
            interpretation=interpretation
        )
    
    def wilcoxon_signed_rank(
        self,
        group1: pd.Series,
        group2: pd.Series,
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """
        Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            group1: First group (paired)
            group2: Second group (paired)
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            StatisticalTest object with rank-biserial correlation
        """
        # Align and clean
        aligned = pd.DataFrame({'g1': group1, 'g2': group2}).dropna()
        
        statistic, p_value = wilcoxon(
            aligned['g1'],
            aligned['g2'],
            alternative=alternative
        )
        
        # Calculate rank-biserial correlation (effect size)
        n = len(aligned)
        r = 1 - (2 * statistic) / (n * (n + 1))
        
        significant = p_value < self.alpha
        
        interpretation = (
            f"{'Significant' if significant else 'No significant'} difference "
            f"(rank-biserial r = {r:.3f})"
        )
        
        return StatisticalTest(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=r,
            interpretation=interpretation
        )
    
    def post_hoc_pairwise(
        self,
        data: pd.DataFrame,
        dv: str,
        within: str,
        subject: str,
        correction: str = 'bonferroni'
    ) -> pd.DataFrame:
        """
        Post-hoc pairwise comparisons with multiple testing correction.
        
        Args:
            data: DataFrame with long-format data
            dv: Dependent variable
            within: Within-subject factor
            subject: Subject identifier
            correction: 'bonferroni' or 'holm'
            
        Returns:
            DataFrame with pairwise comparison results
        """
        groups = data[within].unique()
        results = []
        
        # Perform all pairwise comparisons
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                # Get paired data
                data1 = data[data[within] == g1].set_index(subject)[dv]
                data2 = data[data[within] == g2].set_index(subject)[dv]
                
                # Align
                aligned = pd.DataFrame({'g1': data1, 'g2': data2}).dropna()
                
                # Wilcoxon test
                if len(aligned) > 0:
                    diff = aligned['g1'] - aligned['g2']
                    if np.allclose(diff.values, 0):
                        # No difference at all between paired samples.
                        # Wilcoxon is undefined in this exact case for default zero_method.
                        stat, p_val = 0.0, 1.0
                    else:
                        stat, p_val = wilcoxon(aligned['g1'], aligned['g2'])
                    
                    results.append({
                        'comparison': f"{g1} vs {g2}",
                        'n': len(aligned),
                        'statistic': stat,
                        'p_value': p_val
                    })
        
        results_df = pd.DataFrame(results)
        
        # If no pairwise results, return empty frame with expected columns.
        if results_df.empty:
            for col in ['p_adjusted', 'significant']:
                results_df[col] = []
            return results_df

        # Apply correction
        num_comparisons = len(results_df)
        
        if correction == 'bonferroni':
            results_df['p_adjusted'] = results_df['p_value'] * num_comparisons
            results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)
        elif correction == 'holm':
            # Holm-Bonferroni
            sorted_idx = results_df['p_value'].argsort()
            p_adjusted = []
            for i, idx in enumerate(sorted_idx):
                p_adj = results_df.loc[idx, 'p_value'] * (num_comparisons - i)
                p_adjusted.append(min(p_adj, 1.0))
            results_df['p_adjusted'] = p_adjusted
        
        results_df['significant'] = results_df['p_adjusted'] < self.alpha
        
        return results_df


class EffectSizeCalculator:
    """
    Calculate various effect size measures.
    """
    
    @staticmethod
    def cohens_d(group1: pd.Series, group2: pd.Series, paired: bool = True) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group
            group2: Second group
            paired: Whether samples are paired
            
        Returns:
            Cohen's d value
        """
        if paired:
            aligned = pd.DataFrame({'g1': group1, 'g2': group2}).dropna()
            diff = aligned['g1'] - aligned['g2']
            return diff.mean() / diff.std()
        else:
            mean_diff = group1.mean() - group2.mean()
            pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
            return mean_diff / pooled_std
    
    @staticmethod
    def eta_squared(data: pd.DataFrame, dv: str, factor: str) -> float:
        """
        Calculate eta-squared (η²) effect size for ANOVA.
        
        Args:
            data: DataFrame
            dv: Dependent variable
            factor: Factor variable
            
        Returns:
            η² value (proportion of variance explained)
        """
        # Total sum of squares
        grand_mean = data[dv].mean()
        ss_total = ((data[dv] - grand_mean) ** 2).sum()
        
        # Between-group sum of squares
        ss_between = 0
        for group in data[factor].unique():
            group_data = data[data[factor] == group][dv]
            group_mean = group_data.mean()
            n_group = len(group_data)
            ss_between += n_group * (group_mean - grand_mean) ** 2
        
        return ss_between / ss_total
    
    @staticmethod
    def interpret_effect_size(d: float, measure: str = 'cohens_d') -> str:
        """
        Interpret effect size magnitude.
        
        Args:
            d: Effect size value
            measure: 'cohens_d', 'eta_squared', or 'correlation'
            
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        
        if measure == 'cohens_d':
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        elif measure == 'eta_squared':
            if abs_d < 0.01:
                return "negligible"
            elif abs_d < 0.06:
                return "small"
            elif abs_d < 0.14:
                return "medium"
            else:
                return "large"
        elif measure == 'correlation':
            if abs_d < 0.1:
                return "negligible"
            elif abs_d < 0.3:
                return "small"
            elif abs_d < 0.5:
                return "medium"
            else:
                return "large"
        
        return "unknown"


class VisualizationTools:
    """
    Create publication-quality visualizations.
    """
    
    @staticmethod
    def plot_comparison_boxplot(
        data: pd.DataFrame,
        metric: str,
        group_by: str = 'strategy',
        title: str = None,
        output_file: str = None
    ):
        """
        Create box plot comparing methods.
        
        Args:
            data: DataFrame with results
            metric: Metric column to plot
            group_by: Grouping variable
            title: Plot title
            output_file: Path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        sns.boxplot(data=data, x=group_by, y=metric, palette='Set2')
        
        # Add individual points
        sns.stripplot(data=data, x=group_by, y=metric, 
                     color='black', alpha=0.3, size=3)
        
        plt.title(title or f"{metric} by {group_by}", fontsize=14, fontweight='bold')
        plt.xlabel(group_by.capitalize(), fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_comparison_violin(
        data: pd.DataFrame,
        metric: str,
        group_by: str = 'strategy',
        title: str = None,
        output_file: str = None
    ):
        """
        Create violin plot showing distribution.
        
        Args:
            data: DataFrame with results
            metric: Metric column
            group_by: Grouping variable
            title: Plot title
            output_file: Path to save
        """
        plt.figure(figsize=(10, 6))
        
        sns.violinplot(data=data, x=group_by, y=metric, palette='muted')
        
        plt.title(title or f"{metric} Distribution by {group_by}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel(group_by.capitalize(), fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_correlation_heatmap(
        data: pd.DataFrame,
        metrics: List[str],
        title: str = "Metric Correlations",
        output_file: str = None
    ):
        """
        Create correlation heatmap for metrics.
        
        Args:
            data: DataFrame
            metrics: List of metric columns
            title: Plot title
            output_file: Path to save
        """
        # Calculate correlations
        corr_matrix = data[metrics].corr()
        
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_performance_comparison(
        data: pd.DataFrame,
        metrics: List[str],
        group_by: str = 'strategy',
        title: str = "Performance Comparison",
        output_file: str = None
    ):
        """
        Create multi-metric comparison plot.
        
        Args:
            data: DataFrame
            metrics: List of metrics to compare
            group_by: Grouping variable
            title: Plot title
            output_file: Path to save
        """
        # Calculate means
        means = data.groupby(group_by)[metrics].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        means.T.plot(kind='bar', ax=ax, rot=45)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.legend(title=group_by.capitalize(), loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()


class StatisticalAnalyzer:
    """
    Main analyzer that orchestrates all statistical analyses.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.inferential = InferentialStatistics(alpha=alpha)
        self.descriptives = DescriptiveStatistics()
        self.visualizer = VisualizationTools()
    
    def analyze_experiment_results(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        strategies: List[str],
        output_dir: str = 'analysis_results'
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of experiment results.
        
        Args:
            data: DataFrame with columns: query_id, strategy, metric1, metric2, ...
            metrics: List of metric column names to analyze
            strategies: List of strategy names
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'descriptive_statistics': {},
            'normality_tests': {},
            'comparison_tests': {},
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'visualizations': []
        }
        
        print("=" * 70)
        print("STATISTICAL ANALYSIS OF RETRIEVAL METHODS")
        print("=" * 70)
        
        # 1. Descriptive Statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 70)
        descriptives = self.descriptives.compute_descriptives(data, metrics, 'strategy')
        results['descriptive_statistics'] = descriptives.to_dict('records')
        
        # Save descriptives
        descriptives.to_csv(f'{output_dir}/descriptive_statistics.csv', index=False)
        print(f"✓ Saved: {output_dir}/descriptive_statistics.csv")
        
        # Print summary
        for metric in metrics:
            print(f"\n{metric}:")
            for strategy in strategies:
                row = descriptives[(descriptives['metric'] == metric) & 
                                  (descriptives['strategy'] == strategy)]
                if not row.empty:
                    print(f"  {strategy}: M={row['mean'].values[0]:.4f}, "
                          f"SD={row['std'].values[0]:.4f}, "
                          f"Mdn={row['median'].values[0]:.4f}")
        
        # 2. Normality Tests
        print("\n\n2. NORMALITY TESTS")
        print("-" * 70)
        for metric in metrics:
            results['normality_tests'][metric] = {}
            print(f"\n{metric}:")
            for strategy in strategies:
                strategy_data = data[data['strategy'] == strategy][metric]
                test_result = self.inferential.test_normality(strategy_data)
                results['normality_tests'][metric][strategy] = test_result.to_dict()
                print(f"  {strategy}: p={test_result.p_value:.4f} - {test_result.interpretation}")
        
        # 3. Friedman Test (Overall comparison)
        print("\n\n3. OVERALL COMPARISON (Friedman Test)")
        print("-" * 70)
        for metric in metrics:
            test_result = self.inferential.repeated_measures_anova(
                data, dv=metric, within='strategy', subject='query_id'
            )
            results['comparison_tests'][metric] = test_result.to_dict()
            print(f"\n{metric}:")
            print(f"  χ²={test_result.statistic:.4f}, p={test_result.p_value:.4f}")
            print(f"  {test_result.interpretation}")
        
        # 4. Post-hoc Pairwise Comparisons
        print("\n\n4. POST-HOC PAIRWISE COMPARISONS (Bonferroni corrected)")
        print("-" * 70)
        for metric in metrics:
            pairwise = self.inferential.post_hoc_pairwise(
                data, dv=metric, within='strategy', subject='query_id',
                correction='bonferroni'
            )
            results['pairwise_comparisons'][metric] = pairwise.to_dict('records')
            
            print(f"\n{metric}:")
            for _, row in pairwise.iterrows():
                sig_mark = "***" if row['significant'] else "ns"
                print(f"  {row['comparison']}: p_adj={row['p_adjusted']:.4f} {sig_mark}")
            
            # Save pairwise results
            pairwise.to_csv(f'{output_dir}/pairwise_{metric}.csv', index=False)
        
        # 5. Effect Sizes
        print("\n\n5. EFFECT SIZES")
        print("-" * 70)
        for metric in metrics:
            results['effect_sizes'][metric] = {}
            print(f"\n{metric}:")
            
            # Cohen's d for each pair
            for i, s1 in enumerate(strategies):
                for s2 in strategies[i+1:]:
                    data1 = data[data['strategy'] == s1].set_index('query_id')[metric]
                    data2 = data[data['strategy'] == s2].set_index('query_id')[metric]
                    
                    d = EffectSizeCalculator.cohens_d(data1, data2, paired=True)
                    interp = EffectSizeCalculator.interpret_effect_size(d, 'cohens_d')
                    
                    results['effect_sizes'][metric][f'{s1}_vs_{s2}'] = {
                        'cohens_d': d,
                        'interpretation': interp
                    }
                    
                    print(f"  {s1} vs {s2}: d={d:.3f} ({interp})")
            
            # Eta-squared for overall effect
            eta_sq = EffectSizeCalculator.eta_squared(data, metric, 'strategy')
            results['effect_sizes'][metric]['eta_squared'] = eta_sq
            print(f"  Overall η²={eta_sq:.3f} ({EffectSizeCalculator.interpret_effect_size(eta_sq, 'eta_squared')})")
        
        # 6. Visualizations
        print("\n\n6. CREATING VISUALIZATIONS")
        print("-" * 70)
        for metric in metrics:
            # Box plot
            output_file = f'{output_dir}/boxplot_{metric}.png'
            self.visualizer.plot_comparison_boxplot(
                data, metric, 'strategy',
                title=f"{metric.replace('_', ' ').title()} Comparison",
                output_file=output_file
            )
            results['visualizations'].append(output_file)
            
            # Violin plot
            output_file = f'{output_dir}/violin_{metric}.png'
            self.visualizer.plot_comparison_violin(
                data, metric, 'strategy',
                title=f"{metric.replace('_', ' ').title()} Distribution",
                output_file=output_file
            )
            results['visualizations'].append(output_file)
        
        # Correlation heatmap
        output_file = f'{output_dir}/correlation_heatmap.png'
        self.visualizer.plot_correlation_heatmap(
            data, metrics,
            title="Metric Correlations",
            output_file=output_file
        )
        results['visualizations'].append(output_file)
        
        # Performance comparison
        output_file = f'{output_dir}/performance_comparison.png'
        self.visualizer.plot_performance_comparison(
            data, metrics, 'strategy',
            title="Overall Performance Comparison",
            output_file=output_file
        )
        results['visualizations'].append(output_file)
        
        # Save complete results (convert DataFrames to avoid circular references)
        with open(f'{output_dir}/statistical_analysis_complete.json', 'w') as f:
            # Convert complex objects to serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict('records')
                elif isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"\n✓ Complete analysis saved: {output_dir}/statistical_analysis_complete.json")
        
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("=" * 70)
        
        return results


def main():
    """
    Example usage with synthetic data.
    """
    # Generate example data
    np.random.seed(42)
    n_queries = 100
    
    data_records = []
    for query_id in range(1, n_queries + 1):
        # Simulate three strategies with different performance
        for strategy, (prec_mean, prec_std) in [
            ('keyword', (0.65, 0.15)),
            ('semantic', (0.75, 0.12)),
            ('hybrid', (0.80, 0.10))
        ]:
            data_records.append({
                'query_id': query_id,
                'strategy': strategy,
                'precision': np.clip(np.random.normal(prec_mean, prec_std), 0, 1),
                'recall': np.clip(np.random.normal(prec_mean - 0.05, prec_std), 0, 1),
                'f1_score': np.clip(np.random.normal(prec_mean - 0.02, prec_std), 0, 1),
                'mrr': np.clip(np.random.normal(prec_mean + 0.05, prec_std * 0.8), 0, 1),
            })
    
    data = pd.DataFrame(data_records)
    
    # Run analysis
    analyzer = StatisticalAnalyzer(alpha=0.05)
    results = analyzer.analyze_experiment_results(
        data=data,
        metrics=['precision', 'recall', 'f1_score', 'mrr'],
        strategies=['keyword', 'semantic', 'hybrid'],
        output_dir='example_analysis_results'
    )
    
    print("\nExample analysis complete! Check 'example_analysis_results/' directory")


if __name__ == "__main__":
    main()
