"""
Side-by-Side Comparison Chatbot
================================

Shows answers from all 3 retrieval methods simultaneously.
Allows human evaluation and validation.
Logs results for thesis analysis.

Usage:
    python comparison_chatbot.py
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List

try:
    from blocks.main.retrieval_strategies import retrieve
    from blocks.main.main import (
        extract_bank_codes_from_text,
        parse_currency_hint,
        classify_intents,
        ollama_chat,
    )
except ImportError:
    from retrieval_strategies import retrieve
    from main import (
        extract_bank_codes_from_text,
        parse_currency_hint,
        classify_intents,
        ollama_chat,
    )


def determine_collection(question: str) -> str:
    """Determine which collection to search.
    
    Priority order matters: more specific product types (like FIXED_DEPOSIT)
    should be checked before generic terms (like FX/rate).
    """
    intents = classify_intents(question)
    question_lower = question.lower()
    
    # Check specific product types first (highest priority)
    if "FIXED_DEPOSIT" in intents or "fixed deposit" in question_lower or "fd" in question_lower:
        return "fixed_deposits"
    elif "SAVINGS" in intents or "savings" in question_lower:
        return "savings_accounts"
    elif "LOAN" in intents or any(word in question_lower for word in ["loan", "borrow", "financing", "mortgage"]):
        return "loan"
    elif "CREDIT_CARD" in intents or "card" in question_lower:
        return "credit_cards"
    # Check FX last (lowest priority) since "rate" is too generic
    elif "FX" in intents or any(word in question_lower for word in ["exchange", "forex", "convert"]):
        return "exchange_rates"
    else:
        return "loan"


def get_answer_with_method(question: str, strategy: str) -> Dict[str, Any]:
    """
    Get answer using specific retrieval strategy.
    
    Returns:
        Dictionary with results, answer, and timing
    """
    start_time = time.time()
    
    # Determine collection and filters
    collection = determine_collection(question)
    bank_codes = extract_bank_codes_from_text(question)
    currency = parse_currency_hint(question)
    
    # Retrieve documents
    try:
        results = retrieve(
            collection_name=collection,
            query=question,
            strategy=strategy,
            bank_codes=bank_codes,
            currency=currency,
            limit=5
        )
        retrieval_time = time.time() - start_time
    except Exception as e:
        return {
            "strategy": strategy,
            "error": str(e),
            "retrieval_time": 0,
            "llm_time": 0,
            "total_time": 0
        }
    
    # Format context
    if not results:
        return {
            "strategy": strategy,
            "num_results": 0,
            "answer": "No matching results found.",
            "retrieval_time": retrieval_time,
            "llm_time": 0,
            "total_time": retrieval_time
        }
    
    # Build context for LLM
    context_lines = [f"Found {len(results)} products:\n"]
    
    for i, doc in enumerate(results[:3], 1):
        bank = doc.get("bank", doc.get("bank_name", "Unknown"))
        
        # Get appropriate product name based on collection
        if collection == "exchange_rates":
            product = doc.get("rate_type", "Exchange Rates")
        else:
            product = doc.get("loan_name", doc.get("product_name", doc.get("card_name", "Unknown")))
        
        score = doc.get("_retrieval_score", 0)
        
        # Don't show relevance score to LLM - it's just an internal ranking metric
        context_lines.append(f"{i}. {bank} - {product}")
        
        # Handle collection-specific details
        if collection == "exchange_rates" and "rates" in doc:
            context_lines.append(f"   Base Currency: {doc.get('base_currency', 'N/A')}")
            context_lines.append(f"   Rate Type: {doc.get('rate_type', 'N/A')}")
            context_lines.append(f"   Exchange Rates:")
            for rate in doc["rates"][:10]:  # Show first 10 pairs
                pair = rate.get("pair", "N/A")
                buy = rate.get("buy", "N/A")
                sell = rate.get("sell", "N/A")
                context_lines.append(f"     {pair}: Buy {buy} / Sell {sell}")
            if len(doc["rates"]) > 10:
                context_lines.append(f"     ... and {len(doc['rates']) - 10} more pairs")
        
        elif collection == "fixed_deposits":
            # Show fixed deposit details
            if "deposit_features" in doc:
                features = doc["deposit_features"]
                if "currencies" in features:
                    context_lines.append(f"   Currencies: {', '.join(features['currencies'])}")
                if "initial_deposit" in features:
                    min_dep = features["initial_deposit"]
                    amounts = [f"{curr} {amt}" for curr, amt in min_dep.items()]
                    context_lines.append(f"   Minimum Deposit: {', '.join(amounts)}")
                if "tenors_months" in features:
                    tenors = features["tenors_months"]
                    context_lines.append(f"   Terms Available: {', '.join(map(str, tenors))} months")
            
            # Show interest rate info if available
            if "interest_rate" in doc:
                ir = doc["interest_rate"]
                if isinstance(ir, dict):
                    if "type" in ir:
                        context_lines.append(f"   Interest Rate Type: {ir['type']}")
                    if "effective_from" in ir:
                        context_lines.append(f"   Effective From: {ir['effective_from']}")
                else:
                    context_lines.append(f"   Interest Rate: {ir}")
        
        elif collection == "credit_cards":
            # Show credit card details
            if "network" in doc:
                context_lines.append(f"   Network: {doc['network']}")
            if "category" in doc and doc["category"]:
                categories = doc["category"] if isinstance(doc["category"], list) else [doc["category"]]
                context_lines.append(f"   Categories: {', '.join(categories)}")
            
            # Show fees information
            if "fees" in doc:
                fees = doc["fees"]
                if isinstance(fees, dict):
                    # Annual fees
                    if "annual_fee_principal" in fees:
                        annual_fee = fees["annual_fee_principal"]
                        if isinstance(annual_fee, dict):
                            fee_details = []
                            for tier, amount in annual_fee.items():
                                if tier != "currency":
                                    fee_details.append(f"{tier.replace('_', ' ').title()}: ${amount}")
                            if fee_details:
                                context_lines.append(f"   Annual Fee: {', '.join(fee_details)}")
                    
                    # Interest rates
                    if "purchase_interest_rate" in fees:
                        rate = fees["purchase_interest_rate"]
                        if isinstance(rate, dict) and "rate_percent_per_month" in rate:
                            context_lines.append(f"   Purchase Interest: {rate['rate_percent_per_month']}% per month")
                    
                    if "cash_advance_interest_rate" in fees:
                        rate = fees["cash_advance_interest_rate"]
                        if isinstance(rate, dict) and "rate_percent_per_month" in rate:
                            context_lines.append(f"   Cash Advance Interest: {rate['rate_percent_per_month']}% per month")
            
            # Show features
            if "features" in doc and doc["features"]:
                features_list = doc["features"][:3]  # Show first 3 features
                if features_list:
                    context_lines.append(f"   Key Features: {', '.join(features_list)}")
        
        elif collection == "loan":
            # Show loan details
            if "currency" in doc:
                context_lines.append(f"   Currency: {doc['currency']}")
            if "interest_rate" in doc:
                context_lines.append(f"   Interest Rate: {doc['interest_rate']}")
            if "loan_amount" in doc:
                context_lines.append(f"   Loan Amount: {doc['loan_amount']}")
            if "loan_term" in doc:
                context_lines.append(f"   Loan Term: {doc['loan_term']}")
            if "collateral" in doc:
                context_lines.append(f"   Collateral: {doc['collateral']}")
        
        else:
            # Handle other product types generically
            if "interest_rate" in doc:
                context_lines.append(f"   Interest: {doc['interest_rate']}")
            if "currency" in doc:
                context_lines.append(f"   Currency: {doc['currency']}")
        
        context_lines.append("")
    
    context = "\n".join(context_lines)
    
    # Generate answer with LLM
    llm_start = time.time()
    prompt = f"""Based on these banking products:

{context}

Question: {question}

Provide a brief, helpful answer (2-3 sentences max). Be specific and accurate."""
    
    try:
        answer = ollama_chat([{"role": "user", "content": prompt}])
        llm_time = time.time() - llm_start
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        llm_time = 0
    
    total_time = time.time() - start_time
    
    return {
        "strategy": strategy,
        "collection": collection,
        "num_results": len(results),
        "top_products": [
            {
                "bank": doc.get("bank", doc.get("bank_name", "Unknown")),
                "product": doc.get("loan_name", doc.get("product_name", doc.get("card_name", 
                          doc.get("rate_type", "Unknown")))),
                "score": doc.get("_retrieval_score", 0)
            }
            for doc in results[:3]
        ],
        "answer": answer,
        "retrieval_time": retrieval_time * 1000,  # Convert to ms
        "llm_time": llm_time * 1000,
        "total_time": total_time * 1000
    }


def display_comparison(question: str, results: Dict[str, Dict[str, Any]]):
    """Display results from all 3 methods side-by-side."""
    print("\n" + "="*90)
    print(f"📝 QUESTION: {question}")
    print("="*90)
    
    strategies = ["keyword", "semantic", "hybrid"]
    
    # Show results found
    print("\n📊 RESULTS RETRIEVED:")
    print("-"*90)
    for strategy in strategies:
        result = results[strategy]
        num_results = result.get("num_results", 0)
        time_taken = result.get("retrieval_time", 0)
        print(f"  {strategy.upper():<12} → {num_results} products found ({time_taken:.0f}ms)")
    
    # Show top products
    print("\n🔍 TOP PRODUCTS FOUND:")
    print("-"*90)
    
    for i in range(3):
        print(f"\n  Rank #{i+1}:")
        for strategy in strategies:
            result = results[strategy]
            products = result.get("top_products", [])
            if i < len(products):
                prod = products[i]
                score = prod.get("score", 0)
                print(f"    {strategy.upper():<12} {prod['bank']} - {prod['product'][:40]:<40} (score: {score:.3f})")
            else:
                print(f"    {strategy.upper():<12} (no result)")
    
    # Show answers
    print("\n" + "="*90)
    print("💬 ANSWERS FROM EACH METHOD:")
    print("="*90)
    
    for strategy in strategies:
        result = results[strategy]
        answer = result.get("answer", "No answer")
        total_time = result.get("total_time", 0)
        
        print(f"\n{'─'*90}")
        print(f"🤖 {strategy.upper()} (Total: {total_time:.0f}ms)")
        print(f"{'─'*90}")
        print(f"{answer}\n")
    
    # Show timing summary
    print("="*90)
    print("⏱️  PERFORMANCE:")
    print("-"*90)
    for strategy in strategies:
        result = results[strategy]
        ret_time = result.get("retrieval_time", 0)
        llm_time = result.get("llm_time", 0)
        total_time = result.get("total_time", 0)
        print(f"  {strategy.upper():<12} Retrieval: {ret_time:>6.0f}ms | LLM: {llm_time:>6.0f}ms | Total: {total_time:>6.0f}ms")


def get_human_evaluation() -> Dict[str, Any]:
    """Get human evaluation of the three methods."""
    print("\n" + "="*90)
    print("👤 HUMAN EVALUATION")
    print("="*90)
    print("\nPlease rate the answers from each method:\n")
    
    evaluation = {}
    
    strategies = ["keyword", "semantic", "hybrid"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()}:")
        print("  How would you rate this answer?")
        print("  5 = Excellent | 4 = Good | 3 = Okay | 2 = Poor | 1 = Bad")
        
        while True:
            try:
                rating = input(f"  Rating (1-5): ").strip()
                if rating == "":
                    rating = "3"  # Default
                rating = int(rating)
                if 1 <= rating <= 5:
                    evaluation[strategy] = rating
                    break
                else:
                    print("  Please enter a number between 1 and 5")
            except ValueError:
                print("  Please enter a valid number")
    
    # Ask for best method
    print("\n" + "-"*90)
    print("Which method gave the BEST answer overall?")
    print("  1 = Keyword | 2 = Semantic | 3 = Hybrid")
    
    while True:
        choice = input("Your choice (1-3): ").strip()
        if choice == "1":
            evaluation["best_method"] = "keyword"
            break
        elif choice == "2":
            evaluation["best_method"] = "semantic"
            break
        elif choice == "3":
            evaluation["best_method"] = "hybrid"
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Optional comments
    comments = input("\nOptional comments: ").strip()
    if comments:
        evaluation["comments"] = comments
    
    return evaluation


def save_comparison_log(question: str, results: Dict[str, Dict], evaluation: Dict[str, Any]):
    """Save comparison results to log file for thesis analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "results": results,
        "human_evaluation": evaluation
    }
    
    # Append to log file
    log_file = "comparison_logs.jsonl"
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")
    
    print(f"\n✅ Results saved to {log_file}")


def run_comparison_chatbot():
    """Run the interactive comparison chatbot."""
    print("="*90)
    print("🔬 SIDE-BY-SIDE COMPARISON CHATBOT")
    print("="*90)
    print("\nThis chatbot shows answers from ALL 3 retrieval methods:")
    print("  1️⃣  KEYWORD - Fast, exact matching")
    print("  2️⃣  SEMANTIC - AI-powered understanding") 
    print("  3️⃣  HYBRID - Best of both worlds")
    print("\nYou can compare them and evaluate which is better!")
    print("\nType 'exit', 'quit', or 'q' to stop")
    print("Type 'skip' to skip human evaluation")
    print("="*90)
    
    question_count = 0
    
    while True:
        print("\n" + "─"*90)
        question = input("\n💬 Your Question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['exit', 'quit', 'q']:
            print(f"\n✅ Tested {question_count} questions. Results saved to comparison_logs.jsonl")
            print("👋 Thank you for helping evaluate the retrieval methods!")
            break
        
        if question.lower() == 'skip':
            continue
        
        question_count += 1
        
        # Get answers from all 3 methods
        print("\n🔄 Getting answers from all 3 methods...")
        print("   This may take a moment (especially semantic search)...\n")
        
        results = {}
        for strategy in ["keyword", "semantic", "hybrid"]:
            print(f"   → {strategy}...", end="", flush=True)
            results[strategy] = get_answer_with_method(question, strategy)
            print(" ✓")
        
        # Display comparison
        display_comparison(question, results)
        
        # Get human evaluation
        evaluate = input("\n❓ Would you like to evaluate these answers? (y/n) [y]: ").strip().lower()
        
        if evaluate in ['', 'y', 'yes']:
            evaluation = get_human_evaluation()
            save_comparison_log(question, results, evaluation)
            
            print("\n📊 Your ratings:")
            for strategy in ["keyword", "semantic", "hybrid"]:
                rating = evaluation.get(strategy, "N/A")
                stars = "⭐" * rating if isinstance(rating, int) else ""
                print(f"  {strategy.upper():<12} {stars} ({rating}/5)")
            
            best = evaluation.get("best_method", "N/A")
            print(f"\n🏆 You chose: {best.upper()}")
        else:
            print("\n⏭️  Skipped evaluation")


if __name__ == "__main__":
    try:
        run_comparison_chatbot()
    except KeyboardInterrupt:
        print("\n\n👋 Comparison stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
