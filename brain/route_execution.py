from brain.fast_search import fast_topic_search

_confidence_history = []


def _update_confidence(value: float):
    _confidence_history.append(value)
    if len(_confidence_history) > 100:
        _confidence_history.pop(0)


def _dynamic_threshold():
    if not _confidence_history:
        return None
    return sum(_confidence_history) / len(_confidence_history)


def probe_confidence(query: str) -> float:
    results = fast_topic_search(query)
    print(f"[DEBUG] BM25 results count: {len(results)}")
    if results:
        print(f"[DEBUG] Top score: {results[0].metadata.get('score', 'N/A')}")
    
    scores = [r.metadata.get("score", 0) for r in results[:5]]
    print(f"[DEBUG] All scores: {scores}")
    
    top = scores[0] if scores else 0
    avg = sum(scores) / len(scores) if scores else 0
    confidence = top / (avg + 1e-6)
    print(f"[DEBUG] Confidence: {confidence:.3f}")
    
    _update_confidence(confidence)
    return confidence
def route_execution_mode(query: str) -> str:
    confidence = probe_confidence(query)
    threshold = _dynamic_threshold()

    if threshold is None:
        return "deep"

    if confidence >= threshold:
        return "fast"

    return "deep"