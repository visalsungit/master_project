"""
Retrieval Strategies for Banking Q&A System
============================================

This module implements three retrieval approaches for comparison:
1. Keyword-based retrieval (existing approach)
2. Semantic retrieval using embeddings
3. Hybrid retrieval (combining both)

For Master's thesis: Comparative analysis of retrieval methods
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "banking_db"

# Global model cache (loaded once)
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

# Bank code mappings (fallback for extraction)
BANK_ALIASES = {
    'ppcbank': 'PPCBANK',
    'ppc': 'PPCBANK',
    'aba': 'ABABANK',
    'ababank': 'ABABANK',
    'acleda': 'ACLEDABANK',
    'acledabank': 'ACLEDABANK',
    'canadia': 'CANADIABANK',
    'canadiabank': 'CANADIABANK',
    'wing': 'WINGBANK',
    'wingbank': 'WINGBANK',
    'kbprasac': 'KBPRASACBANK',
    'kbprasacbank': 'KBPRASACBANK',
    'prasac': 'KBPRASACBANK',
    'kb prasac': 'KBPRASACBANK',
    'kb': 'KBPRASACBANK',  # Only when combined with other context
    'ftb': 'FTBBANK',
    'ftbbank': 'FTBBANK',
    'lolc': 'LOLCMFI',
    'chief': 'CHIEFBANK',
    'chiefbank': 'CHIEFBANK',
    'sathapana': 'SATHAPANABANK',
    'sathapanabank': 'SATHAPANABANK',
}

# Bank code normalization - handle data inconsistencies across collections
# Maps extracted codes to actual codes used in product collections
BANK_CODE_NORMALIZATION = {
    'PPCBBANK': ['PPCBANK', 'PPCBBANK'],  # bank_code collection uses PPCBBANK, but loan uses PPCBANK
    'PPCBANK': ['PPCBANK', 'PPCBBANK'],
    'KBPRASACBANK': ['KBPRASAC', 'KBPRASACBANK'],  # Handle KB PRASAC variations
    'KBPRASAC': ['KBPRASAC', 'KBPRASACBANK'],
    'KB': ['KBPRASAC', 'KBPRASACBANK'],  # KB alone should map to KB PRASAC
}


def normalize_bank_codes(bank_codes: List[str]) -> List[str]:
    """
    Normalize bank codes to handle data inconsistencies across collections.
    
    Some banks have different codes in bank_code collection vs product collections.
    This function expands codes to include all known variations.
    
    Args:
        bank_codes: List of bank codes to normalize
        
    Returns:
        Expanded list including all variations
    """
    if not bank_codes:
        return bank_codes
    
    expanded = []
    for code in bank_codes:
        if code in BANK_CODE_NORMALIZATION:
            # Add all variations
            expanded.extend(BANK_CODE_NORMALIZATION[code])
        else:
            # Keep original
            expanded.append(code)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for code in expanded:
        if code not in seen:
            seen.add(code)
            result.append(code)
    
    return result


def extract_bank_codes_fallback(query: str) -> List[str]:
    """
    Fallback bank code extraction using simple pattern matching.
    Used when main.py's extract_bank_codes_from_text fails.
    
    Args:
        query: User query text
        
    Returns:
        List of matched bank codes
    """
    query_lower = query.lower()
    
    # Special handling for multi-word bank names
    # Check for "kb prasac" first (before tokenizing)
    if 'kb prasac' in query_lower or 'kbprasac' in query_lower:
        return ['KBPRASACBANK']
    
    # Also check for just "prasac"
    if 'prasac' in query_lower:
        return ['KBPRASACBANK']
    
    tokens = set(re.findall(r'\b[a-z0-9]+\b', query_lower))
    
    found_codes = []
    for alias, code in BANK_ALIASES.items():
        if alias in tokens or alias in query_lower:
            # Skip generic 'kb' unless it's alone with prasac-related terms
            if alias == 'kb' and ('prasac' not in query_lower and 'deposit' not in query_lower):
                continue
            if code not in found_codes:
                found_codes.append(code)
    
    return found_codes


def get_embedding_model() -> SentenceTransformer:
    """
    Get or load the sentence transformer model (singleton pattern).
    
    Returns:
        SentenceTransformer model for generating embeddings
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def compute_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def keyword_retrieval(
    collection_name: str,
    query: str,
    bank_codes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Strategy 1: Keyword-based retrieval (baseline approach).
    
    Uses MongoDB queries with exact/partial matching on bank_code, currency, etc.
    This is your current implementation.
    
    Args:
        collection_name: MongoDB collection to search
        query: User query (used for extracting filters)
        bank_codes: List of bank codes to filter by
        currency: Currency to filter by
        limit: Maximum number of results
    
    Returns:
        List of matching documents with metadata
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    mongo_query: Dict[str, Any] = {}
    
    # Apply filters - use fallback extraction if no bank codes provided
    if not bank_codes or len(bank_codes) == 0:
        bank_codes = extract_bank_codes_fallback(query)
    
    # Normalize bank codes to handle data inconsistencies
    if bank_codes:
        bank_codes = normalize_bank_codes(bank_codes)
        # BUGFIX: MongoDB $in operator not working correctly - use $or instead
        if len(bank_codes) == 1:
            mongo_query["bank_code"] = bank_codes[0]
        else:
            mongo_query["$or"] = [{"bank_code": code} for code in bank_codes]
    
    if currency:
        # Handle different currency field structures
        ccy_upper = currency.upper()
        if collection_name == "loan":
            # BUGFIX: Use $or instead of $in
            currency_query = {"$or": [{"currency": ccy_upper}, {"currency": currency.lower()}]}
            if "$or" in mongo_query:
                # Combine with existing $or (from bank_codes)
                mongo_query = {"$and": [mongo_query, currency_query]}
            else:
                mongo_query.update(currency_query)
        elif collection_name == "exchange_rates":
            mongo_query["base_currency"] = ccy_upper
        else:
            # For fixed_deposits, savings_accounts, credit_cards
            currency_query = {"$or": [
                {"currency": ccy_upper},
                {"currency": currency.lower()},
                {"currencies_supported": ccy_upper},
                {"deposit_features.currencies": ccy_upper}  # For fixed_deposits
            ]}
            if "$or" in mongo_query:
                # Combine with existing $or (from bank_codes)
                mongo_query = {"$and": [mongo_query, currency_query]}
            else:
                mongo_query.update(currency_query)
    
    results = list(collection.find(mongo_query, {"_id": 0}).limit(limit))
    
    # Add retrieval metadata
    for i, doc in enumerate(results):
        doc["_retrieval_method"] = "keyword"
        doc["_retrieval_rank"] = i + 1
        doc["_retrieval_score"] = 1.0 - (i * 0.05)  # Simple ranking score
    
    client.close()
    return results


def semantic_retrieval(
    collection_name: str,
    query: str,
    bank_codes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Strategy 2: Semantic retrieval using embeddings.
    
    Computes query embedding and finds documents with highest cosine similarity.
    Does NOT use keyword filters - pure semantic search.
    
    Args:
        collection_name: MongoDB collection to search
        query: User query text
        bank_codes: Optional filter (applied after semantic ranking)
        currency: Optional filter (applied after semantic ranking)
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of documents ranked by semantic similarity
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[collection_name]
    
    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()
    
    # Fetch all documents with embeddings
    mongo_query = {"embedding": {"$exists": True}}
    documents = list(collection.find(mongo_query, {"_id": 0}))
    
    if not documents:
        client.close()
        return []
    
    # Compute similarity scores
    scored_docs = []
    for doc in documents:
        doc_embedding = doc.get("embedding", [])
        if not doc_embedding:
            continue
        
        similarity = compute_cosine_similarity(query_embedding, doc_embedding)
        
        if similarity >= similarity_threshold:
            doc["_retrieval_method"] = "semantic"
            doc["_retrieval_score"] = similarity
            scored_docs.append(doc)
    
    # Sort by similarity (descending)
    scored_docs.sort(key=lambda x: x["_retrieval_score"], reverse=True)
    
    # Apply post-filters if specified
    if bank_codes:
        # Normalize bank codes for filtering
        normalized_codes = normalize_bank_codes(bank_codes)
        scored_docs = [doc for doc in scored_docs if doc.get("bank_code") in normalized_codes]
    
    if currency:
        ccy_upper = currency.upper()
        scored_docs = [
            doc for doc in scored_docs 
            if (ccy_upper in str(doc.get("currency", "")).upper() or 
                ccy_upper in str(doc.get("currencies_supported", "")).upper() or
                ccy_upper in str(doc.get("deposit_features", {}).get("currencies", [])))
        ]
    
    # Add ranking metadata
    for i, doc in enumerate(scored_docs[:limit]):
        doc["_retrieval_rank"] = i + 1
    
    client.close()
    return scored_docs[:limit]


def hybrid_retrieval(
    collection_name: str,
    query: str,
    bank_codes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 10,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Strategy 3: Hybrid retrieval (keyword + semantic).
    
    Combines keyword and semantic approaches with weighted scoring:
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    
    Args:
        collection_name: MongoDB collection to search
        query: User query text
        bank_codes: List of bank codes to filter by
        currency: Currency to filter by
        limit: Maximum number of results
        alpha: Weight for semantic score (0-1). 
               0.5 = equal weight, 1.0 = pure semantic, 0.0 = pure keyword
    
    Returns:
        List of documents ranked by hybrid score
    """
    # Get results from both methods
    keyword_results = keyword_retrieval(collection_name, query, bank_codes, currency, limit=50)
    semantic_results = semantic_retrieval(collection_name, query, bank_codes, currency, limit=50)
    
    # Create document map by unique identifier
    def get_doc_id(doc: Dict[str, Any]) -> str:
        """Create unique ID for document."""
        bank = doc.get("bank_code", doc.get("bank", "unknown"))
        name = doc.get("product_name", doc.get("loan_name", doc.get("card_name", "unknown")))
        return f"{bank}_{name}"
    
    # Normalize scores
    keyword_map = {}
    for doc in keyword_results:
        doc_id = get_doc_id(doc)
        # Keyword score: position-based (first result = 1.0)
        keyword_score = doc.get("_retrieval_score", 0.5)
        keyword_map[doc_id] = (doc, keyword_score)
    
    semantic_map = {}
    for doc in semantic_results:
        doc_id = get_doc_id(doc)
        # Semantic score: already computed (cosine similarity)
        semantic_score = doc.get("_retrieval_score", 0.0)
        semantic_map[doc_id] = (doc, semantic_score)
    
    # Combine scores
    all_doc_ids = set(keyword_map.keys()) | set(semantic_map.keys())
    hybrid_results = []
    
    for doc_id in all_doc_ids:
        # Get scores (default 0 if not in one method)
        keyword_doc, keyword_score = keyword_map.get(doc_id, (None, 0.0))
        semantic_doc, semantic_score = semantic_map.get(doc_id, (None, 0.0))
        
        # Use document from whichever method found it
        doc = keyword_doc if keyword_doc else semantic_doc
        
        # Compute hybrid score
        hybrid_score = (alpha * semantic_score) + ((1 - alpha) * keyword_score)
        
        doc["_retrieval_method"] = "hybrid"
        doc["_retrieval_score"] = hybrid_score
        doc["_keyword_score"] = keyword_score
        doc["_semantic_score"] = semantic_score
        
        hybrid_results.append(doc)
    
    # Sort by hybrid score
    hybrid_results.sort(key=lambda x: x["_retrieval_score"], reverse=True)
    
    # Add ranking
    for i, doc in enumerate(hybrid_results[:limit]):
        doc["_retrieval_rank"] = i + 1
    
    return hybrid_results[:limit]


def retrieve(
    collection_name: str,
    query: str,
    strategy: str = "keyword",
    bank_codes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 10,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Unified retrieval interface for all strategies.
    
    Args:
        collection_name: MongoDB collection to search
        query: User query text
        strategy: Retrieval method - "keyword", "semantic", or "hybrid"
        bank_codes: List of bank codes to filter by
        currency: Currency to filter by
        limit: Maximum number of results
        **kwargs: Additional parameters (e.g., alpha for hybrid)
    
    Returns:
        List of retrieved documents with scores
    
    Example:
        >>> results = retrieve("loan", "personal loan USD", strategy="semantic")
        >>> results = retrieve("loan", "personal loan USD", strategy="hybrid", alpha=0.7)
    """
    if strategy == "keyword":
        return keyword_retrieval(collection_name, query, bank_codes, currency, limit)
    elif strategy == "semantic":
        return semantic_retrieval(collection_name, query, bank_codes, currency, limit)
    elif strategy == "hybrid":
        alpha = kwargs.get("alpha", 0.5)
        return hybrid_retrieval(collection_name, query, bank_codes, currency, limit, alpha)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'keyword', 'semantic', or 'hybrid'")


def format_results_for_llm(results: List[Dict[str, Any]], max_results: int = 5) -> str:
    """
    Format retrieval results for LLM consumption.
    
    Args:
        results: List of retrieved documents
        max_results: Maximum results to include
    
    Returns:
        Formatted string for LLM context
    """
    if not results:
        return "No matching products found."
    
    lines = []
    for i, doc in enumerate(results[:max_results], 1):
        bank = doc.get("bank", doc.get("bank_name", "Unknown"))
        product = doc.get("loan_name", doc.get("product_name", doc.get("card_name", "Product")))
        score = doc.get("_retrieval_score", 0.0)
        method = doc.get("_retrieval_method", "unknown")
        
        lines.append(f"{i}. {bank} - {product}")
        lines.append(f"   Retrieval: {method} (score: {score:.3f})")
        
        # Add key details
        if "interest_rate" in doc:
            lines.append(f"   Interest Rate: {doc['interest_rate']}")
        if "currency" in doc:
            lines.append(f"   Currency: {doc['currency']}")
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing retrieval strategies...\n")
    
    test_query = "I need a personal loan in USD with low interest rate"
    
    print(f"Query: {test_query}\n")
    print("="*60)
    
    # Test keyword
    print("\n1. KEYWORD RETRIEVAL:")
    kw_results = retrieve("loan", test_query, strategy="keyword", currency="USD", limit=3)
    print(f"   Found {len(kw_results)} results")
    for r in kw_results[:3]:
        print(f"   - {r.get('bank', 'N/A')}: {r.get('loan_name', 'N/A')} (score: {r.get('_retrieval_score', 0):.3f})")
    
    # Test semantic
    print("\n2. SEMANTIC RETRIEVAL:")
    sem_results = retrieve("loan", test_query, strategy="semantic", limit=3)
    print(f"   Found {len(sem_results)} results")
    for r in sem_results[:3]:
        print(f"   - {r.get('bank', 'N/A')}: {r.get('loan_name', 'N/A')} (score: {r.get('_retrieval_score', 0):.3f})")
    
    # Test hybrid
    print("\n3. HYBRID RETRIEVAL:")
    hyb_results = retrieve("loan", test_query, strategy="hybrid", limit=3, alpha=0.5)
    print(f"   Found {len(hyb_results)} results")
    for r in hyb_results[:3]:
        print(f"   - {r.get('bank', 'N/A')}: {r.get('loan_name', 'N/A')} (score: {r.get('_retrieval_score', 0):.3f})")
    
    print("\n" + "="*60)
    print("✅ Retrieval strategies module ready!")
