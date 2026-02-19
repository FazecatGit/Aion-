from typing import List, Optional


def calculate_precision(retrieved: List[str], relevant: List[str]) -> float:
    if not retrieved:
        return 0.0
    
    relevant_set = set(relevant)
    matches = sum(1 for doc in retrieved if doc in relevant_set)
    return matches / len(retrieved)


def calculate_recall(retrieved: List[str], relevant: List[str]) -> float:
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    matches = sum(1 for doc in retrieved if doc in relevant_set)
    return matches / len(relevant)


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_retrieval(
    retrieved: List[str],
    relevant: List[str],
) -> dict:
    precision = calculate_precision(retrieved, relevant)
    recall = calculate_recall(retrieved, relevant)
    f1 = calculate_f1(precision, recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
