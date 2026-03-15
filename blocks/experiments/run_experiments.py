"""
Automated Experiment Runner
============================

This module orchestrates the complete experimental workflow:
1. Load experiment configuration
2. Generate or load test queries
3. Run retrieval experiments for all strategies
4. Collect results with timing metrics
5. Perform statistical analysis
6. Generate comprehensive reports

Usage:
    python run_experiments.py --config experiment_config.json
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import project modules
try:
    from blocks.main.retrieval_strategies import retrieve
    from blocks.experiments.evaluation_framework import RetrievalEvaluator
    from blocks.experiments.experimental_design import ExperimentConfig
    from blocks.experiments.statistical_analysis import StatisticalAnalyzer
except ImportError:
    from main.retrieval_strategies import retrieve
    from evaluation_framework import RetrievalEvaluator
    from experimental_design import ExperimentConfig
    from statistical_analysis import StatisticalAnalyzer


class ExperimentRunner:
    """
    Manages the complete experimental workflow.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config: ExperimentConfig object
        """
        self.config = config
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def load_test_queries(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test queries from file.
        
        Args:
            filepath: Path to test queries JSON file
            
        Returns:
            List of test query dictionaries
        """
        print(f"\nLoading test queries from: {filepath}")
        with open(filepath, 'r') as f:
            queries = json.load(f)
        print(f"✓ Loaded {len(queries)} test queries")
        return queries

    def _default_test_query_files(self) -> Dict[str, Path]:
        """Resolve default query files for each collection.

        Returns a mapping of collection name -> JSON file path.
        """
        base_dir = Path(__file__).resolve().parent

        # Note: the loan file in this repo is named with a space: "test_queries _loan.json".
        # We support both spellings to be robust.
        loan_candidates = [
            base_dir / 'test_queries_loan.json',
            base_dir / 'test_queries _loan.json',
        ]
        loan_path = next((p for p in loan_candidates if p.exists()), loan_candidates[-1])

        return {
            'loan': loan_path,
            'credit_cards': base_dir / 'test_queries_credit_card.json',
            'fixed_deposits': base_dir / 'test_queries_fixed_deposits.json',
            'savings_accounts': base_dir / 'test_queries_saving_accounts.json',
        }

    def _validate_test_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and lightly normalize test query records."""
        if not isinstance(queries, list):
            raise ValueError("Test queries JSON must be a list of objects")

        validated: List[Dict[str, Any]] = []
        for idx, q in enumerate(queries):
            if not isinstance(q, dict):
                raise ValueError(f"Test query at index {idx} is not an object")
            if 'query' not in q or 'collection' not in q:
                raise ValueError(
                    f"Test query at index {idx} missing required keys: 'query' and/or 'collection'"
                )
            validated.append(q)

        return validated

    def generate_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from the 4 curated JSON files in this folder.

        This replaces the old (missing) generator behavior and ensures the
        experiment runner always uses the same fixed test set.
        """
        print("\nLoading test queries from curated JSON files...")

        files_by_collection = self._default_test_query_files()
        combined: List[Dict[str, Any]] = []

        for collection in self.config.collections:
            if collection not in files_by_collection:
                raise ValueError(
                    f"No default test query file configured for collection: {collection}"
                )
            path = files_by_collection[collection]
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing test queries file for collection '{collection}': {path}"
                )

            queries = self.load_test_queries(str(path))
            queries = self._validate_test_queries(queries)

            # Sanity check: file contents should match its collection.
            mismatched = [q for q in queries if q.get('collection') != collection]
            if mismatched:
                raise ValueError(
                    f"{path.name} contains queries for other collections; expected '{collection}'. "
                    f"Example mismatch: {mismatched[0].get('collection')}"
                )

            combined.extend(queries)

        print(f"✓ Combined {len(combined)} total test queries across {len(self.config.collections)} collections")
        return combined
    
    def run_single_query(
        self,
        query: str,
        collection: str,
        strategy: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run a single query with specified strategy.
        
        Args:
            query: Query text
            collection: Collection name
            strategy: Retrieval strategy ('keyword', 'semantic', 'hybrid')
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with results and metrics
        """
        try:
            start_time = time.time()
            
            # Call retrieve function
            result = retrieve(
                collection_name=collection,
                query=query,
                strategy=strategy,
                limit=top_k
            )
            
            retrieval_time = time.time() - start_time
            
            # Extract results - retrieve() returns a list of documents
            return {
                'query': query,
                'collection': collection,
                'strategy': strategy,
                'num_results': len(result) if isinstance(result, list) else 0,
                'retrieval_time': retrieval_time,
                'products': result if isinstance(result, list) else [],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'query': query,
                'collection': collection,
                'strategy': strategy,
                'num_results': 0,
                'retrieval_time': 0,
                'products': [],
                'success': False,
                'error': str(e)
            }
    
    def run_experiments(
        self,
        test_queries: List[Dict[str, Any]],
        save_intermediate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run experiments for all queries and strategies.
        
        Args:
            test_queries: List of test query dictionaries
            save_intermediate: Save results after each batch
            
        Returns:
            List of all experiment results
        """
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENTS")
        print("=" * 70)
        
        self.start_time = datetime.now()
        
        total_experiments = len(test_queries) * len(self.config.strategies)
        print(f"\nTotal experiments: {total_experiments}")
        print(f"Queries: {len(test_queries)}")
        print(f"Strategies: {', '.join(self.config.strategies)}")
        print(f"Top-K: {self.config.top_k}")
        
        results = []
        
        # Progress bar
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        for query_dict in test_queries:
            query = query_dict['query']
            collection = query_dict['collection']
            query_id = query_dict.get('id', f"q_{len(results)}")
            
            for strategy in self.config.strategies:
                # Run single experiment
                result = self.run_single_query(
                    query=query,
                    collection=collection,
                    strategy=strategy,
                    top_k=self.config.top_k
                )
                
                # Add metadata
                result['query_id'] = query_id
                result['difficulty'] = query_dict.get('difficulty', 'unknown')
                result['timestamp'] = datetime.now().isoformat()
                
                results.append(result)
                pbar.update(1)
                
                # Brief delay to avoid overwhelming the system
                time.sleep(0.1)
            
            # Save intermediate results every 10 queries
            if save_intermediate and len(results) % (10 * len(self.config.strategies)) == 0:
                self._save_intermediate_results(results)
        
        pbar.close()
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n✓ Experiments completed in {duration:.2f} seconds")
        print(f"  Average time per experiment: {duration / total_experiments:.2f}s")
        
        # Count successes and failures
        successes = sum(1 for r in results if r['success'])
        failures = sum(1 for r in results if not r['success'])
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        
        if failures > 0:
            print("\n⚠ Some experiments failed. Check experiment_results_raw.json for details.")
        
        self.results = results
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to file."""
        with open('experiment_results_intermediate.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def save_results(self, output_file: str = 'experiment_results_raw.json'):
        """
        Save raw experiment results to file.
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'results': self.results
            }, f, indent=2, default=str)
        
        print(f"\n✓ Raw results saved to: {output_file}")
    
    def compute_evaluation_metrics(
        self,
        test_dataset: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute evaluation metrics for all experiments.
        
        If test_dataset with ground truth is provided, computes precision/recall/etc.
        Otherwise, only reports retrieval statistics.
        
        Args:
            test_dataset: Optional test dataset with ground truth
            
        Returns:
            DataFrame with metrics for all experiments
        """
        print("\n" + "=" * 70)
        print("COMPUTING EVALUATION METRICS")
        print("=" * 70)
        
        if test_dataset:
            # Use evaluation framework with ground truth
            print("\nUsing ground truth for precision/recall calculation...")
            evaluator = RetrievalEvaluator(test_dataset)
            
            # Organize results by query and strategy
            for result in self.results:
                if result['success']:
                    evaluator.results[result['strategy']].append({
                        'query': result['query'],
                        'products': result['products'],
                        'retrieval_time': result['retrieval_time']
                    })
            
            # Compute metrics
            comparison_results = evaluator.compare_strategies()

            # Flatten average metrics to DataFrame (one row per strategy)
            rows = []
            for strategy, result in comparison_results.items():
                row = {
                    'strategy': strategy,
                    'k': result.get('k'),
                    'num_queries': result.get('num_queries')
                }
                row.update(result.get('avg_metrics', {}))
                rows.append(row)

            metrics_df = pd.DataFrame(rows)
            
        else:
            # Basic statistics without ground truth
            print("\nComputing basic retrieval statistics (no ground truth)...")
            
            metrics_records = []
            for result in self.results:
                if result['success']:
                    metrics_records.append({
                        'query_id': result['query_id'],
                        'query': result['query'],
                        'collection': result['collection'],
                        'strategy': result['strategy'],
                        'difficulty': result['difficulty'],
                        'num_results': result['num_results'],
                        'retrieval_time': result['retrieval_time'],
                        'has_results': result['num_results'] > 0
                    })
            
            metrics_df = pd.DataFrame(metrics_records)
        
        # Save metrics
        metrics_df.to_csv('experiment_metrics.csv', index=False)
        print(f"\n✓ Metrics saved to: experiment_metrics.csv")
        
        # Print summary
        print("\nMetrics Summary:")
        if 'strategy' in metrics_df.columns:
            summary_cols = [c for c in ['num_results', 'retrieval_time', 'latency_ms',
                                        'precision@5', 'recall@5', 'f1@5', 'mrr', 'ndcg@5']
                            if c in metrics_df.columns]
            if summary_cols:
                summary = metrics_df.groupby('strategy')[summary_cols].mean(numeric_only=True)
                print(summary)
            else:
                print("No numeric summary columns found.")
        
        return metrics_df
    
    def run_statistical_analysis(
        self,
        metrics_df: pd.DataFrame,
        metrics_to_analyze: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on results.
        
        Args:
            metrics_df: DataFrame with computed metrics
            metrics_to_analyze: List of metric column names to analyze
            
        Returns:
            Dictionary with statistical analysis results
        """
        if metrics_to_analyze is None:
            # Determine which metrics are available
            potential_metrics = ['precision', 'recall', 'f1_score', 'mrr', 'ndcg', 
                                'num_results', 'retrieval_time']
            metrics_to_analyze = [m for m in potential_metrics if m in metrics_df.columns]
        
        if not metrics_to_analyze:
            print("\n⚠ No metrics available for statistical analysis")
            return {}
        
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)
        print(f"\nAnalyzing metrics: {', '.join(metrics_to_analyze)}")
        
        # Check if we have required columns
        if 'query_id' not in metrics_df.columns or 'strategy' not in metrics_df.columns:
            print("\n⚠ Cannot perform statistical analysis: missing query_id or strategy columns")
            return {}
        
        # Run statistical analysis
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        analysis_results = analyzer.analyze_experiment_results(
            data=metrics_df,
            metrics=metrics_to_analyze,
            strategies=self.config.strategies,
            output_dir='statistical_analysis_results'
        )
        
        return analysis_results


def main():
    """
    Main function to run complete experimental workflow.
    """
    parser = argparse.ArgumentParser(
        description='Run retrieval methods experiments with statistical analysis'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiment_config.json',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--queries',
        type=str,
        default=None,
        help='Path to test queries file (optional, will generate if not provided)'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='Path to ground truth dataset for evaluation (optional)'
    )
    parser.add_argument(
        '--skip-experiments',
        action='store_true',
        help='Skip running experiments, only analyze existing results'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AUTOMATED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"\nConfiguration: {args.config}")
    
    # Load or create configuration
    if os.path.exists(args.config):
        print(f"✓ Loading configuration from: {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        print(f"⚠ Configuration file not found, creating default...")
        from experimental_design import ExperimentDesigner
        designer = ExperimentDesigner()
        config = designer.design_experiment()
        config.save(args.config)
        print(f"✓ Default configuration saved to: {args.config}")
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Load or generate test queries
    if args.queries:
        queries_path = Path(args.queries)
        if queries_path.is_dir():
            # Load all matching files in a directory (useful for custom query sets).
            candidates = sorted(queries_path.glob('test_queries*.json'))
            if not candidates:
                raise FileNotFoundError(f"No test_queries*.json files found in: {queries_path}")
            test_queries = []
            for p in candidates:
                test_queries.extend(runner._validate_test_queries(runner.load_test_queries(str(p))))
            print(f"✓ Loaded {len(test_queries)} total test queries from directory")
        elif queries_path.exists():
            test_queries = runner._validate_test_queries(runner.load_test_queries(str(queries_path)))
        else:
            raise FileNotFoundError(f"--queries path not found: {queries_path}")
    else:
        test_queries = runner.generate_test_queries()
    
    # Run experiments (unless skipped)
    if not args.skip_experiments:
        results = runner.run_experiments(test_queries, save_intermediate=True)
        runner.save_results()
    else:
        print("\n⚠ Skipping experiments (--skip-experiments flag)")
        # Try to load existing results
        if os.path.exists('experiment_results_raw.json'):
            with open('experiment_results_raw.json', 'r') as f:
                data = json.load(f)
                runner.results = data['results']
            print("✓ Loaded existing results from: experiment_results_raw.json")
        else:
            print("✗ No existing results found!")
            return
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        print(f"\n✓ Loading ground truth from: {args.ground_truth}")
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
    
    # Compute metrics
    metrics_df = runner.compute_evaluation_metrics(test_dataset=ground_truth)
    
    # Statistical analysis
    if len(metrics_df) > 0 and 'strategy' in metrics_df.columns:
        analysis_results = runner.run_statistical_analysis(metrics_df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. experiment_results_raw.json - Raw experiment results")
    print("  2. experiment_metrics.csv - Computed metrics")
    print("  3. statistical_analysis_results/ - Statistical analysis outputs")
    print("     - descriptive_statistics.csv")
    print("     - pairwise_*.csv (for each metric)")
    print("     - *.png (visualizations)")
    print("     - statistical_analysis_complete.json")
    print("\nNext Steps:")
    print("  1. Review visualizations in statistical_analysis_results/")
    print("  2. Generate final report: python report_generator.py")
    print("  3. Include results in thesis document")
    print("=" * 70)


if __name__ == "__main__":
    main()
