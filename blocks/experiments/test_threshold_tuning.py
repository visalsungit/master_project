"""
Threshold Tuning Experiment
============================

Test different similarity thresholds to find optimal value.
This experiment is REQUIRED for thesis methodology validation.

Usage:
    python test_threshold_tuning.py
"""

import json

try:
    from blocks.main.retrieval_strategies import semantic_retrieval
except ImportError:
    from main.retrieval_strategies import semantic_retrieval

# Test queries
TEST_QUERIES = [
    {"query": "personal loan for emergency", "expected_collection": "loan"},
    {"query": "ABA credit card rewards", "expected_collection": "credit_cards"},
    {"query": "high interest savings account", "expected_collection": "savings_accounts"},
    {"query": "home loan with low rate", "expected_collection": "loan"},
    {"query": "USD to KHR exchange rate", "expected_collection": "exchange_rates"},
]

# Thresholds to test
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_threshold(threshold: float) -> dict:
    """Test a specific threshold value."""
    print(f"\n{'='*70}")
    print(f"Testing Threshold: {threshold}")
    print(f"{'='*70}")
    
    results = {
        "threshold": threshold,
        "total_results": 0,
        "avg_results_per_query": 0,
        "queries": []
    }
    
    for test in TEST_QUERIES:
        query = test["query"]
        collection = test["expected_collection"]
        
        # Get results with this threshold
        docs = semantic_retrieval(
            collection_name=collection,
            query=query,
            similarity_threshold=threshold,
            limit=10
        )
        
        # Analyze results
        num_results = len(docs)
        avg_score = sum(d.get("_retrieval_score", 0) for d in docs) / num_results if num_results > 0 else 0
        
        query_result = {
            "query": query,
            "num_results": num_results,
            "avg_score": avg_score,
            "top_3_scores": [d.get("_retrieval_score", 0) for d in docs[:3]]
        }
        
        results["queries"].append(query_result)
        results["total_results"] += num_results
        
        print(f"\nQuery: {query}")
        print(f"  Results: {num_results}, Avg Score: {avg_score:.3f}")
        if docs:
            print(f"  Top 3 scores: {[f'{s:.3f}' for s in query_result['top_3_scores']]}")
    
    results["avg_results_per_query"] = results["total_results"] / len(TEST_QUERIES)
    
    return results


def compare_thresholds():
    """Compare all thresholds and recommend best value."""
    print("\n" + "="*70)
    print("THRESHOLD TUNING EXPERIMENT")
    print("="*70)
    print("\nTesting thresholds:", THRESHOLDS)
    
    all_results = []
    
    for threshold in THRESHOLDS:
        result = test_threshold(threshold)
        all_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Avg Results':<15} {'Total Results'}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['threshold']:<12.1f} {result['avg_results_per_query']:<15.1f} {result['total_results']}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Find threshold with good balance (not too few, not too many results)
    # Target: 2-5 results per query on average
    best_threshold = None
    best_score = float('inf')
    
    for result in all_results:
        avg = result['avg_results_per_query']
        # Penalize if too few (< 2) or too many (> 5)
        if avg < 2:
            penalty = abs(3 - avg)  # Want closer to 3
        elif avg > 5:
            penalty = abs(3 - avg)
        else:
            penalty = abs(3 - avg)  # Sweet spot around 3 results
        
        if penalty < best_score:
            best_score = penalty
            best_threshold = result['threshold']
    
    print(f"\n✅ Recommended threshold: {best_threshold}")
    print(f"   Reasoning: Provides ~3 results per query on average")
    print(f"   - Too low threshold → too many irrelevant results")
    print(f"   - Too high threshold → miss relevant results")
    print(f"   - {best_threshold} → good balance")
    
    # Save results for thesis
    with open("threshold_tuning_results.json", "w") as f:
        json.dump({
            "all_results": all_results,
            "recommendation": best_threshold,
            "test_queries": TEST_QUERIES
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: threshold_tuning_results.json")
    print("   Use this data in your thesis methodology section!")


if __name__ == "__main__":
    print("Starting threshold tuning experiment...")
    print("This will test multiple threshold values to find the optimal setting.")
    print("This experiment is important for your thesis validation!\n")
    
    input("Press Enter to start...")
    
    compare_thresholds()
    
    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)
    print("\n📝 Next steps for thesis:")
    print("1. Review threshold_tuning_results.json")
    print("2. Document this experiment in your Methodology chapter")
    print("3. Explain why you chose the recommended threshold")
    print("4. Include this as part of your experimental design")
