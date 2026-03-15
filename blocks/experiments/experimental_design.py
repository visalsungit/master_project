"""
Experimental Design Framework for Retrieval Methods Comparison
===============================================================

This module provides tools for designing rigorous experiments to compare
the three retrieval strategies (keyword, semantic, hybrid) for research purposes.

Features:
- Sample size calculation
- Experiment configuration
- Test query generation
- Baseline establishment
- Reproducibility controls
"""

import json
import math
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for retrieval experiments."""
    name: str
    description: str
    strategies: List[str]  # ['keyword', 'semantic', 'hybrid']
    collections: List[str]  # ['loan', 'credit_cards', 'fixed_deposits', 'savings_accounts']
    num_queries_per_collection: int
    top_k: int  # Number of results to retrieve
    random_seed: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class SampleSizeCalculator:
    """
    Calculate required sample size for statistical power analysis.
    
    Based on standard formulas for comparing means with paired samples
    (since we're comparing methods on the same queries).
    """
    
    @staticmethod
    def calculate_sample_size(
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.80,
        num_groups: int = 3
    ) -> int:
        """
        Calculate minimum sample size for detecting differences between retrieval methods.
        
        Args:
            effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
            alpha: Significance level (typically 0.05)
            power: Statistical power (typically 0.80 or 0.90)
            num_groups: Number of methods being compared
            
        Returns:
            Minimum number of test queries needed
        """
        # Z-scores for alpha and beta
        z_alpha = 1.96  # For alpha=0.05 (two-tailed)
        z_beta = 0.84   # For power=0.80
        
        if power == 0.90:
            z_beta = 1.28
        elif power == 0.95:
            z_beta = 1.645
        
        # Formula for paired samples (more efficient than independent)
        # n = 2 * ((z_alpha + z_beta) / effect_size)^2
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        # Adjust for multiple comparisons (Bonferroni correction)
        if num_groups > 2:
            n = n * (1 + 0.1 * (num_groups - 2))
        
        return math.ceil(n)
    
    @staticmethod
    def report_power_analysis(effect_sizes: List[float] = None) -> Dict[str, Any]:
        """
        Generate a power analysis report for different scenarios.
        
        Returns:
            Dictionary with sample size recommendations
        """
        if effect_sizes is None:
            effect_sizes = [0.2, 0.3, 0.5, 0.8]  # small, small-medium, medium, large
        
        report = {
            "description": "Sample size requirements for comparing 3 retrieval methods",
            "parameters": {
                "alpha": 0.05,
                "power": 0.80,
                "num_methods": 3,
                "test_type": "paired (repeated measures)"
            },
            "recommendations": {}
        }
        
        for effect in effect_sizes:
            n = SampleSizeCalculator.calculate_sample_size(effect_size=effect)
            effect_label = "small" if effect <= 0.3 else "medium" if effect <= 0.6 else "large"
            report["recommendations"][f"effect_{effect}_{effect_label}"] = {
                "effect_size": effect,
                "min_queries": n,
                "suggested_queries": n + 10  # Add buffer
            }
        
        # Add practical recommendation
        report["practical_recommendation"] = {
            "minimum": 50,  # Absolute minimum for meaningful statistics
            "recommended": 100,  # Better for detecting medium effects
            "ideal": 200,  # Excellent for detecting small-medium effects
            "per_collection": 25  # If testing 4 collections
        }
        
        return report


class ExperimentDesigner:
    """
    Design and configure complete experiments.
    """
    
    def __init__(self, name: str = "Retrieval Methods Comparison"):
        self.name = name
    
    def design_experiment(
        self,
        num_queries_per_collection: int = 25,
        collections: List[str] = None,
        random_seed: int = 42
    ) -> ExperimentConfig:
        """
        Design a complete experiment with appropriate parameters.
        
        Args:
            num_queries_per_collection: Queries per collection
            collections: Collections to test (default: all)
            random_seed: Random seed for reproducibility
            
        Returns:
            ExperimentConfig object
        """
        if collections is None:
            collections = ['loan', 'credit_cards', 'fixed_deposits', 'savings_accounts']
        
        config = ExperimentConfig(
            name=self.name,
            description=f"Comparative evaluation of keyword, semantic, and hybrid retrieval strategies",
            strategies=['keyword', 'semantic', 'hybrid'],
            collections=collections,
            num_queries_per_collection=num_queries_per_collection,
            top_k=5,
            random_seed=random_seed,
            timestamp=datetime.now().isoformat()
        )
        
        return config
    
    def create_baseline_config(self) -> Dict[str, Any]:
        """
        Create baseline configuration for comparison.
        
        Returns:
            Baseline metrics and parameters
        """
        return {
            "baseline_method": "keyword",
            "baseline_description": "Traditional keyword-based retrieval using text matching",
            "comparison_methods": ["semantic", "hybrid"],
            "primary_metrics": [
                "precision@5",
                "recall@5", 
                "mrr",
                "ndcg@5",
                "f1_score"
            ],
            "secondary_metrics": [
                "retrieval_time",
                "llm_time",
                "total_time"
            ],
            "hypotheses": {
                "H1": "Semantic retrieval will achieve higher precision than keyword-based retrieval",
                "H2": "Hybrid retrieval will achieve the best overall performance across all metrics",
                "H3": "Semantic retrieval will be slower than keyword-based retrieval",
                "H4": "There will be no significant difference in LLM response time across methods"
            }
        }


def main():
    """
    Run experiment design: calculate sample size and create configuration.
    Note: Test queries should be created manually in test_queries.json
    """
    print("=" * 70)
    print("EXPERIMENTAL DESIGN FOR RETRIEVAL METHODS COMPARISON")
    print("=" * 70)
    
    # 1. Power Analysis
    print("\n1. POWER ANALYSIS - Sample Size Calculation")
    print("-" * 70)
    calculator = SampleSizeCalculator()
    power_report = calculator.report_power_analysis()
    
    print(f"\nParameters:")
    print(f"  - Significance level (α): {power_report['parameters']['alpha']}")
    print(f"  - Statistical power: {power_report['parameters']['power']}")
    print(f"  - Number of methods: {power_report['parameters']['num_methods']}")
    print(f"  - Test type: {power_report['parameters']['test_type']}")
    
    print(f"\nSample Size Requirements:")
    for key, rec in power_report['recommendations'].items():
        print(f"  - Effect size {rec['effect_size']} ({key.split('_')[-1]}): {rec['min_queries']} queries (suggested: {rec['suggested_queries']})")
    
    print(f"\nPractical Recommendation:")
    prac = power_report['practical_recommendation']
    print(f"  - Minimum: {prac['minimum']} total queries")
    print(f"  - Recommended: {prac['recommended']} total queries")
    print(f"  - Ideal: {prac['ideal']} total queries")
    print(f"  - Per collection: {prac['per_collection']} queries (for 4 collections)")
    
    # Save power analysis
    with open('power_analysis_report.json', 'w') as f:
        json.dump(power_report, f, indent=2)
    print(f"\n✓ Power analysis saved to: power_analysis_report.json")
    
    # 2. Design Experiment Configuration
    print("\n\n2. EXPERIMENT CONFIGURATION")
    print("-" * 70)
    designer = ExperimentDesigner(name="Master's Thesis - Retrieval Comparison")
    
    # Use recommended sample size
    config = designer.design_experiment(
        num_queries_per_collection=25,  # 100 total queries for 4 collections
        random_seed=42
    )
    
    print(f"\nExperiment Configuration:")
    print(f"  - Name: {config.name}")
    print(f"  - Strategies: {', '.join(config.strategies)}")
    print(f"  - Collections: {', '.join(config.collections)}")
    print(f"  - Queries per collection: {config.num_queries_per_collection}")
    print(f"  - Total queries needed: {len(config.collections) * config.num_queries_per_collection}")
    print(f"  - Top-K results: {config.top_k}")
    print(f"  - Random seed: {config.random_seed}")
    
    # Save configuration
    config.save('experiment_config.json')
    print(f"\n✓ Experiment config saved to: experiment_config.json")
    
    # 3. Baseline Configuration
    print("\n\n3. BASELINE AND HYPOTHESES")
    print("-" * 70)
    baseline_config = designer.create_baseline_config()
    
    print(f"\nBaseline Method: {baseline_config['baseline_method']}")
    print(f"Description: {baseline_config['baseline_description']}")
    print(f"\nComparison Methods: {', '.join(baseline_config['comparison_methods'])}")
    
    print(f"\nPrimary Metrics:")
    for metric in baseline_config['primary_metrics']:
        print(f"  - {metric}")
    
    print(f"\nSecondary Metrics:")
    for metric in baseline_config['secondary_metrics']:
        print(f"  - {metric}")
    
    print(f"\nResearch Hypotheses:")
    for h_id, hypothesis in baseline_config['hypotheses'].items():
        print(f"  {h_id}: {hypothesis}")
    
    # Save baseline config
    with open('baseline_config.json', 'w') as f:
        json.dump(baseline_config, f, indent=2)
    print(f"\n✓ Baseline config saved to: baseline_config.json")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("EXPERIMENT DESIGN COMPLETE")
    print("=" * 70)
    print(f"\nFiles Created:")
    print(f"  1. power_analysis_report.json - Sample size calculations")
    print(f"  2. experiment_config.json - Experiment parameters")
    print(f"  3. baseline_config.json - Baseline and hypotheses")
    print(f"\n⚠️  IMPORTANT: Create test_queries.json manually")
    print(f"   The file must have this structure:")
    print(f"   [")
    print(f"     {{\"query\": \"your query\", \"collection\": \"loan\", \"relevant_docs\": [...]}}")
    print(f"   ]")
    print(f"   Aim for {prac['recommended']} queries total with ground truth.")
    print(f"\nNext Steps:")
    print(f"  1. Create test_queries.json with your queries and ground truth")
    print(f"  2. Run experiments: python run_experiments.py --queries test_queries.json")
    print(f"  3. Analyze results: python statistical_analysis.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
