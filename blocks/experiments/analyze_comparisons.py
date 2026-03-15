"""
Analysis Tool for Comparison Logs
==================================

Analyzes human evaluation data from comparison_chatbot.py
Generates statistics for thesis.

Usage:
    python analyze_comparisons.py
"""
# Run 3 mondels for comparison analysis
import json
from collections import defaultdict
from typing import List, Dict, Any


def load_comparison_logs(filename: str = "comparison_logs.jsonl") -> List[Dict]:
    """Load comparison logs from file."""
    logs = []
    try:
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ No log file found: {filename}")
        return []
    
    return logs


def analyze_logs(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze comparison logs and generate statistics."""
    if not logs:
        return {}
    
    # Count evaluations
    total_questions = len(logs)
    
    # Rating statistics
    ratings = defaultdict(list)
    best_method_counts = defaultdict(int)
    
    for log in logs:
        eval_data = log.get("human_evaluation", {})
        
        # Collect ratings
        for strategy in ["keyword", "semantic", "hybrid"]:
            if strategy in eval_data:
                ratings[strategy].append(eval_data[strategy])
        
        # Count best method votes
        best = eval_data.get("best_method")
        if best:
            best_method_counts[best] += 1
    
    # Calculate averages
    avg_ratings = {}
    for strategy, rating_list in ratings.items():
        if rating_list:
            avg_ratings[strategy] = sum(rating_list) / len(rating_list)
    
    # Performance statistics
    avg_times = defaultdict(list)
    avg_results_found = defaultdict(list)
    
    for log in logs:
        results = log.get("results", {})
        for strategy, data in results.items():
            if "total_time" in data:
                avg_times[strategy].append(data["total_time"])
            if "num_results" in data:
                avg_results_found[strategy].append(data["num_results"])
    
    avg_time_stats = {
        strategy: sum(times) / len(times) if times else 0
        for strategy, times in avg_times.items()
    }
    
    avg_results_stats = {
        strategy: sum(counts) / len(counts) if counts else 0
        for strategy, counts in avg_results_found.items()
    }
    
    return {
        "total_questions": total_questions,
        "avg_ratings": avg_ratings,
        "best_method_votes": dict(best_method_counts),
        "avg_response_time_ms": avg_time_stats,
        "avg_results_found": avg_results_stats,
        "all_ratings": dict(ratings)
    }


def print_analysis(stats: Dict[str, Any]):
    """Print analysis results in a nice format."""
    print("\n" + "="*70)
    print("📊 COMPARISON ANALYSIS REPORT")
    print("="*70)
    
    total = stats.get("total_questions", 0)
    print(f"\n📝 Total Questions Evaluated: {total}")
    
    if total == 0:
        print("\n⚠️  No data to analyze yet. Run comparison_chatbot.py first!")
        return
    
    # Human ratings
    print("\n" + "─"*70)
    print("⭐ AVERAGE HUMAN RATINGS (1-5 scale)")
    print("─"*70)
    
    avg_ratings = stats.get("avg_ratings", {})
    for strategy in ["keyword", "semantic", "hybrid"]:
        rating = avg_ratings.get(strategy, 0)
        stars = "⭐" * int(round(rating))
        print(f"  {strategy.upper():<12} {rating:.2f}/5  {stars}")
    
    # Best method votes
    print("\n" + "─"*70)
    print("🏆 BEST METHOD VOTES")
    print("─"*70)
    
    votes = stats.get("best_method_votes", {})
    total_votes = sum(votes.values())
    
    for strategy in ["keyword", "semantic", "hybrid"]:
        count = votes.get(strategy, 0)
        percentage = (count / total_votes * 100) if total_votes > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"  {strategy.upper():<12} {count:>3} votes ({percentage:>5.1f}%)  {bar}")
    
    # Performance
    print("\n" + "─"*70)
    print("⏱️  AVERAGE RESPONSE TIME")
    print("─"*70)
    
    times = stats.get("avg_response_time_ms", {})
    for strategy in ["keyword", "semantic", "hybrid"]:
        time_ms = times.get(strategy, 0)
        print(f"  {strategy.upper():<12} {time_ms:>7.0f}ms")
    
    # Results found
    print("\n" + "─"*70)
    print("🔍 AVERAGE RESULTS FOUND")
    print("─"*70)
    
    results_found = stats.get("avg_results_found", {})
    for strategy in ["keyword", "semantic", "hybrid"]:
        count = results_found.get(strategy, 0)
        print(f"  {strategy.upper():<12} {count:>5.1f} products")
    
    # Winner summary
    print("\n" + "="*70)
    print("🎯 SUMMARY")
    print("="*70)
    
    # Determine winners
    best_rating_method = max(avg_ratings.items(), key=lambda x: x[1])[0] if avg_ratings else "N/A"
    best_voted_method = max(votes.items(), key=lambda x: x[1])[0] if votes else "N/A"
    fastest_method = min(times.items(), key=lambda x: x[1])[0] if times else "N/A"
    
    print(f"  Highest Rating:    {best_rating_method.upper()}")
    print(f"  Most Voted:        {best_voted_method.upper()}")
    print(f"  Fastest Response:  {fastest_method.upper()}")
    
    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if best_voted_method == "hybrid":
        print("  ✅ HYBRID method is preferred by users!")
        print("     Combines speed and accuracy effectively.")
    elif best_voted_method == "semantic":
        print("  ✅ SEMANTIC method is preferred by users!")
        print("     Users value understanding over speed.")
    else:
        print("  ✅ KEYWORD method is preferred by users!")
        print("     Users value speed and exact matching.")


def export_for_thesis(stats: Dict[str, Any], output_file: str = "thesis_data.json"):
    """Export data in format ready for thesis."""
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Data exported to {output_file} for thesis analysis")


if __name__ == "__main__":
    print("Loading comparison logs...")
    logs = load_comparison_logs()
    
    if logs:
        print(f"✅ Loaded {len(logs)} comparison sessions\n")
        stats = analyze_logs(logs)
        print_analysis(stats)
        
        # Export
        export = input("\n📤 Export data for thesis? (y/n) [y]: ").strip().lower()
        if export in ['', 'y', 'yes']:
            export_for_thesis(stats)
    else:
        print("\n⚠️  No comparison data found.")
        print("Run: python comparison_chatbot.py")
        print("to start collecting evaluation data!")
