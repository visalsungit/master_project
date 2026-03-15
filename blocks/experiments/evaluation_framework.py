"""
Evaluation Framework for Retrieval Strategies
==============================================

This module implements metrics and evaluation tools for comparing
the three retrieval approaches for your Master's thesis.

Metrics implemented:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- F1 Score
"""

import json
import time
import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    from blocks.main.retrieval_strategies import retrieve
except ImportError:
    from main.retrieval_strategies import retrieve


class RetrievalEvaluator:
    """
    Evaluates retrieval strategies on test queries with ground truth.
    """
    
    def __init__(self, test_dataset: List[Dict[str, Any]]):
        """
        Initialize evaluator with test dataset.
        
        Args:
            test_dataset: List of test cases with format:
                [
                    {
                        "query": "personal loan USD",
                        "collection": "loan",
                        "relevant_docs": ["ABABANK_Personal Loan", "ACLEDABANK_Personal Loan"],
                        "bank_codes": ["ABABANK"],
                        "currency": "USD"
                    },
                    ...
                ]
        """
        self.test_dataset = test_dataset
        self.results = defaultdict(list)
    
    def _doc_id(self, doc: Dict[str, Any]) -> str:
        """Create unique document identifier."""
        bank = doc.get("bank_code", doc.get("bank", "unknown"))
        name = doc.get("product_name", doc.get("loan_name", doc.get("card_name", "unknown")))
        return f"{bank}_{name}"
    
    def _dedupe_preserve_order(self, lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Precision@K: Proportion of retrieved docs that are relevant.
        
        Formula: |retrieved ∩ relevant| / k
        """
        retrieved_k = self._dedupe_preserve_order(retrieved)[:k]
        relevant_set = set(relevant)
        
        if len(retrieved_k) == 0:
            return 0.0
        
        return len([r for r in retrieved_k if r in relevant_set]) / k  # or use / max(1, len(retrieved_k)) if you prefer
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Recall@K: Proportion of relevant docs that are retrieved.
        
        Formula: |retrieved ∩ relevant| / |relevant|
        """
        retrieved_k = set(self._dedupe_preserve_order(retrieved)[:k])
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0.0
        
        return len(retrieved_k & relevant_set) / len(relevant_set)
    
    def f1_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K.
        
        Formula: 2 * (P * R) / (P + R)
        """
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        
        if p + r == 0:
            return 0.0
        
        return 2 * (p * r) / (p + r)
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        MRR: Reciprocal of the rank of the first relevant document.
        
        Formula: 1 / rank_of_first_relevant
        """
        relevant_set = set(relevant)
        for i, doc_id in enumerate(self._dedupe_preserve_order(retrieved), 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Considers order of relevant documents.
        
        Formula: DCG@K / IDCG@K
        """
        retrieved_k = self._dedupe_preserve_order(retrieved)[:k]
        relevant_set = set(relevant)

        # DCG (binary relevance)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, 1):
            rel = 1.0 if doc_id in relevant_set else 0.0
            dcg += rel / math.log2(i + 1)

        # IDCG: best possible ordering (all relevant items at top)
        ideal_rels = [1.0] * min(k, len(relevant_set))
        idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal_rels, 1))

        return 0.0 if idcg == 0 else dcg / idcg
    
    def evaluate_query(
        self,
        query: str,
        collection: str,
        relevant_docs: List[str],
        strategy: str,
        k: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Returns:
            Dictionary of metric scores
        """
        # Retrieve documents
        start_time = time.time()
        results = retrieve(
            collection, 
            query, 
            strategy=strategy,
            limit=k,
            **kwargs
        )
        latency = time.time() - start_time
        
        # Extract document IDs
        retrieved_ids = [self._doc_id(doc) for doc in results]
        
        # Compute metrics
        metrics = {
            f"precision@{k}": self.precision_at_k(retrieved_ids, relevant_docs, k),
            f"recall@{k}": self.recall_at_k(retrieved_ids, relevant_docs, k),
            f"f1@{k}": self.f1_at_k(retrieved_ids, relevant_docs, k),
            "mrr": self.mean_reciprocal_rank(retrieved_ids, relevant_docs),
            f"ndcg@{k}": self.ndcg_at_k(retrieved_ids, relevant_docs, k),
            "latency_ms": latency * 1000
        }
        
        return metrics
    
    def evaluate_strategy(
        self,
        strategy: str,
        k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy across all test queries.
        
        Args:
            strategy: "keyword", "semantic", or "hybrid"
            k: Number of results to consider
            **kwargs: Strategy-specific parameters
        
        Returns:
            Aggregated metrics and per-query results
        """
        print(f"\nEvaluating {strategy.upper()} strategy...")
        
        all_metrics = defaultdict(list)
        query_results = []
        
        for i, test_case in enumerate(self.test_dataset, 1):
            query = test_case["query"]
            collection = test_case["collection"]
            relevant = test_case["relevant_docs"]
            
            # Get optional filters
            filters = {
                "bank_codes": test_case.get("bank_codes"),
                "currency": test_case.get("currency")
            }
            filters.update(kwargs)
            
            # Evaluate
            metrics = self.evaluate_query(
                query, collection, relevant, strategy, k, **filters
            )
            
            # Store results
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
            
            query_results.append({
                "query": query,
                "metrics": metrics
            })
            
            print(f"  Query {i}/{len(self.test_dataset)}: {query[:50]}... ", end="")
            print(f"P@{k}={metrics[f'precision@{k}']:.3f}, MRR={metrics['mrr']:.3f}")
        
        # Compute averages
        avg_metrics = {
            metric_name: sum(values) / len(values)
            for metric_name, values in all_metrics.items()
        }
        
        return {
            "strategy": strategy,
            "k": k,
            "num_queries": len(self.test_dataset),
            "avg_metrics": avg_metrics,
            "query_results": query_results
        }
    
    def compare_strategies(
        self,
        strategies: List[str] = ["keyword", "semantic", "hybrid"],
        k: int = 5,
        hybrid_alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies side-by-side.
        
        Returns:
            Comparison results with rankings
        """
        print(f"\n{'='*60}")
        print(f"COMPARING RETRIEVAL STRATEGIES")
        print(f"{'='*60}")
        print(f"Test queries: {len(self.test_dataset)}")
        print(f"K (top results): {k}")
        
        comparison_results = {}
        
        for strategy in strategies:
            kwargs = {}
            if strategy == "hybrid":
                kwargs["alpha"] = hybrid_alpha
            
            result = self.evaluate_strategy(strategy, k, **kwargs)
            comparison_results[strategy] = result
        
        # Print comparison table
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        metrics_to_show = [f"precision@{k}", f"recall@{k}", f"f1@{k}", "mrr", f"ndcg@{k}", "latency_ms"]
        
        # Header
        print(f"{'Metric':<20}", end="")
        for strategy in strategies:
            print(f"{strategy.upper():<15}", end="")
        print()
        print("-" * (20 + 15 * len(strategies)))
        
        # Rows
        for metric in metrics_to_show:
            print(f"{metric:<20}", end="")
            for strategy in strategies:
                value = comparison_results[strategy]["avg_metrics"][metric]
                if metric == "latency_ms":
                    print(f"{value:<15.2f}", end="")
                else:
                    print(f"{value:<15.4f}", end="")
            print()
        
        print()
        
        # Determine winner for each metric
        print("Best Strategy per Metric:")
        for metric in metrics_to_show:
            if metric == "latency_ms":
                continue  # Skip latency for winner
            
            best_strategy = max(
                strategies,
                key=lambda s: comparison_results[s]["avg_metrics"][metric]
            )
            best_value = comparison_results[best_strategy]["avg_metrics"][metric]
            print(f"  {metric:<20} → {best_strategy.upper()} ({best_value:.4f})")
        
        return comparison_results


def create_sample_test_dataset() -> List[Dict[str, Any]]:
    """
    Create a sample test dataset for evaluation.
    
    In your thesis, you should manually create a larger dataset (50-100 queries)
    with verified ground truth relevant documents.
    """
    return [
        {
            "query": "personal loan with low interest rate in USD",
            "collection": "loan",
            "relevant_docs": [
                "ABABANK_ABA Personal Loan",
                "ACLEDABANK_Personal Loan",
            ],
            "currency": "USD"
        },
        {
            "query": "credit card with no annual fee",
            "collection": "credit_cards",
            "relevant_docs": [
                "ABABANK_Mastercard Standard",
            ],
            "currency": "USD"
        },
        {
            "query": "fixed deposit with monthly interest payment",
            "collection": "fixed_deposits",
            "relevant_docs": [
                "ACLEDABANK_Term Deposit",
            ]
        },
        {
            "query": "best USD to KHR exchange rate",
            "collection": "exchange_rates",
            "relevant_docs": [
                "ABABANK_spot_exchange_rate",
                "ACLEDABANK_spot_exchange_rate",
            ]
        },
        {
            "query": "savings account with high interest",
            "collection": "savings_accounts",
            "relevant_docs": [
                "ACLEDABANK_Savings Account",
            ]
        }
    ]


if __name__ == "__main__":
    print("Evaluation Framework Demo\n")
    print("="*60)
    
    # Create sample dataset
    test_data = create_sample_test_dataset()
    print(f"Loaded {len(test_data)} test queries")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(test_data)
    
    # Compare all strategies
    results = evaluator.compare_strategies(
        strategies=["keyword", "semantic", "hybrid"],
        k=5,
        hybrid_alpha=0.5
    )
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("\nTo use in your thesis:")
    print("1. Create comprehensive test dataset (50-100 queries)")
    print("2. Manually verify ground truth relevant documents")
    print("3. Run evaluation: python evaluation_framework.py")
    print("4. Analyze results for statistical significance")
