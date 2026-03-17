import os
import logging
from pathlib import Path

from brain.timed_llm import TimedLLM

def _env_bool(name: str, default: str = "0") -> bool:
	return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data"))
#FAISS_DIR = os.getenv("RAG_FAISS_DIR", str(BASE_DIR / "faiss_index"))
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "cache" / "chroma_db"))
INDEX_META_PATH = os.getenv("RAG_INDEX_META_PATH", str(Path(CHROMA_DIR) / "index_meta.json"))

# Embedding model settings
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")

# LLM settings
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3")  # qwen3 for general/code reasoning; qwen3.5 was slow on RAG retrieval tasks
MATH_LLM_MODEL = os.getenv("RAG_MATH_LLM_MODEL", "qwen3")  # Dedicated math model for tutor; empty = fall back to LLM_MODEL
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0"))

# Fallback stronger model — used after 1 failed attempt on hard problems.
# Set to a frontier model name available via Ollama (e.g. "qwen3:32b", "deepseek-coder-v2", "codellama:34b")
# or leave empty to disable model escalation.
FALLBACK_LLM_MODEL = os.getenv("RAG_FALLBACK_LLM_MODEL", "")

# LLM context / generation limits (adjust based on model VRAM)
LLM_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "16384"))          # context window tokens
LLM_NUM_PREDICT = int(os.getenv("RAG_LLM_NUM_PREDICT", "8192"))   # max output tokens
LLM_TIMEOUT_SECONDS = int(os.getenv("RAG_LLM_TIMEOUT", "180"))    # per-call timeout (seconds)


LLM_NUM_CTX_HARD = int(os.getenv("RAG_LLM_NUM_CTX_HARD", "32768"))
LLM_NUM_PREDICT_HARD = int(os.getenv("RAG_LLM_NUM_PREDICT_HARD", "16384"))


def make_llm(model: str = None, temperature: float = None, *,
             num_ctx: int = None, num_predict: int = None):
    """Shared LLM factory. Returns a TimedLLM wrapper with logging/diagnostics.
    
    Sets num_ctx and num_predict to prevent unbounded context/generation.
    Pass explicit num_ctx/num_predict to override the defaults (e.g. for hard problems).
    Automatically disables thinking mode on qwen3.x models.
    """
    from langchain_ollama import OllamaLLM
    m = model if model is not None else LLM_MODEL
    t = temperature if temperature is not None else LLM_TEMPERATURE
    ctx = num_ctx if num_ctx is not None else LLM_NUM_CTX
    pred = num_predict if num_predict is not None else LLM_NUM_PREDICT
    # qwen3 models have 'thinking' on by default — disable via reasoning=False
    reasoning = False if "qwen3" in m.lower() else None
    raw_llm = OllamaLLM(
        model=m,
        temperature=t,
        reasoning=reasoning,
        num_ctx=ctx,
        num_predict=pred,
    )
    return TimedLLM(raw_llm)

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))

# Text splitting settings
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

# Hybrid fusion settings
FUSION_MODE = os.getenv("RAG_FUSION_MODE", "rrf")  # "rrf" or "weighted"
FUSION_ALPHA = float(os.getenv("RAG_FUSION_ALPHA", "0.5"))  # For weighted mode: 1.0 = semantic, 0.0 = keyword
FUSION_K_PARAM = int(os.getenv("RAG_FUSION_K_PARAM", "60"))  # For RRF mode: higher = more conservative

# Query enhancement settings
ENABLE_QUERY_SPELL_CORRECTION = _env_bool("RAG_ENABLE_QUERY_SPELL_CORRECTION", "1")
ENABLE_QUERY_REWRITE = _env_bool("RAG_ENABLE_QUERY_REWRITE", "0")
ENABLE_QUERY_EXPANSION = _env_bool("RAG_ENABLE_QUERY_EXPANSION", "0")

# Retrieval depth settings
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_RETRIEVAL_CANDIDATE_MULTIPLIER", "3"))

# Reranking settings
RERANK_METHOD = os.getenv("RAG_RERANK_METHOD", "keyword")  # "none", "keyword", "cross_encoder"
CROSS_ENCODER_MODEL = os.getenv("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE = 32 # For cross-encoder reranking

# ChromaDB config
CHROMA_PERSIST_DIR = CHROMA_DIR		      # where vectors live
CHROMA_COLLECTION_NAME = "aion-code"     # collection for code docs
CHROMA_CHUNK_SIZE = 512                 # optional, for new ingest

# ─── Language Maps (shared across agent modules) ───────────────────────────────

# Human-readable language names
LANG_NAMES: dict[str, str] = {
    '.py': 'Python', '.go': 'Go', '.cpp': 'C++', '.c': 'C',
    '.rs': 'Rust', '.js': 'JavaScript', '.ts': 'TypeScript',
    '.java': 'Java', '.cs': 'C#', '.rb': 'Ruby', '.php': 'PHP',
    '.swift': 'Swift', '.kt': 'Kotlin',
}

# Markdown code-fence language identifiers
LANG_FENCE: dict[str, str] = {
    '.py': 'python', '.go': 'go', '.cpp': 'cpp', '.c': 'c',
    '.rs': 'rust', '.js': 'javascript', '.ts': 'typescript',
    '.java': 'java', '.cs': 'csharp', '.rb': 'ruby', '.php': 'php',
    '.swift': 'swift', '.kt': 'kotlin',
}

# Compiler / syntax-check commands per extension
LANG_CHECK_CMD: dict[str, list[str]] = {
    '.go':  ['go', 'build'],
    '.rs':  ['rustc', '--edition', '2021'],
    '.cpp': ['g++', '-fsyntax-only'],
    '.c':   ['gcc', '-fsyntax-only'],
    '.ts':  ['tsc', '--noEmit'],
}

# Lint commands per extension (deeper analysis beyond basic compilation)
# These are only used when the binary is found on PATH.
LANG_LINT_CMD: dict[str, list[str]] = {
    '.go':   ['go', 'vet'],
    '.py':   ['ruff', 'check', '--select', 'E,F,W'],
    '.cpp':  ['clang-tidy'],
    '.c':    ['clang-tidy'],
    '.rs':   ['cargo', 'clippy', '--message-format=short', '--quiet'],
    '.java': ['javac', '-Xlint:all'],
    '.ts':   ['npx', 'eslint'],
    '.js':   ['npx', 'eslint'],
}

# Query enhancement keywords for RAG search
LANG_QUERY_ENHANCEMENT: dict[str, str] = {
    '.py': 'Python programming', '.go': 'Go Golang programming',
    '.cpp': 'C++ programming', '.c': 'C programming',
    '.rs': 'Rust programming', '.js': 'JavaScript programming',
    '.ts': 'TypeScript programming', '.java': 'Java programming',
    '.cs': 'C# programming', '.rb': 'Ruby programming',
    '.php': 'PHP programming', '.swift': 'Swift programming',
    '.kt': 'Kotlin programming',
}

# Extension → RAG topic labels (used to build context-aware topic filters).
# Always combines with ["algorithms", "clean-code"] for code-editing tasks.
EXT_TO_TOPIC: dict[str, str] = {
    '.py': 'python', '.go': 'go', '.cpp': 'cpp', '.c': 'cpp',
    '.rs': 'rust', '.js': 'web', '.ts': 'web',
    '.java': 'java', '.kt': 'java',
}

# Document source keywords for language-aware chunk filtering
LANG_DOC_KEYWORDS: dict[str, list[str]] = {
    '.py':   ['python'],
    '.go':   ['go', 'golang'],
    '.cpp':  ['c++', 'cplusplus', 'stroustrup', 'effective_modern', 'concurrency_in_action'],
    '.c':    ['c_programming', 'c++', 'cplusplus'],
    '.rs':   ['rust', 'rustaceans'],
    '.ts':   ['typescript', 'angular'],
    '.js':   ['javascript', 'typescript', 'angular'],
    '.java': ['java'],
    '.cs':   ['c#', 'csharp'],
    '.rb':   ['ruby'],
}

# Keywords that indicate a doc is for the WRONG language
LANG_IRRELEVANT_DOC_KEYWORDS: dict[str, list[str]] = {
    '.py':  ['c++', 'cplusplus', 'stroustrup', 'golang', 'go_programming', 'rust', 'blender', 'angular', 'typescript', 'swift', 'kotlin'],
    '.go':  ['python', 'c++', 'cplusplus', 'stroustrup', 'rust', 'blender', 'angular', 'typescript', 'swift', 'kotlin'],
    '.cpp': ['python', 'golang', 'go_programming', 'rust', 'blender', 'angular', 'typescript', 'swift', 'kotlin'],
    '.c':   ['python', 'golang', 'go_programming', 'rust', 'blender', 'angular', 'typescript', 'swift', 'kotlin'],
    '.rs':  ['python', 'golang', 'go_programming', 'c++', 'cplusplus', 'blender', 'angular', 'typescript', 'swift', 'kotlin'],
    '.ts':  ['python', 'golang', 'go_programming', 'c++', 'cplusplus', 'rust', 'blender', 'swift', 'kotlin'],
    '.js':  ['python', 'golang', 'go_programming', 'c++', 'cplusplus', 'rust', 'blender', 'swift', 'kotlin'],
}

# Language-agnostic docs always pass through filters
# These match against source filenames (after normalising spaces/hyphens to _)
UNIVERSAL_DOC_KEYWORDS: list[str] = [
    # Software-design classics
    'clean-code', 'clean_code', 'refactoring', 'legacy', 'design', 'design_patterns',
    # Algorithm / DS textbooks & concepts
    'algorithm', 'data_structure', 'data-structure', 'leetcode', 'competitive',
    'introduction_to_algo', 'clrs', 'skiena', 'sedgewick', 'cormen',
    'cracking_the_coding', 'programming_pearls', 'art_of_programming',
    'elements_of_programming', 'competitive_programming',
    # Math / theory
    'math', 'discrete', 'combinatorics', 'number_theory', 'probability',
    'linear_algebra', 'calculus',
    # Generic reference markers
    'handbook', 'textbook', 'tutorial', 'reference', 'cheat_sheet',
    'editorial', 'solution_explanation',
]

# Topic labels (from ingest metadata) that are language-agnostic
UNIVERSAL_TOPICS: set[str] = {
    'algorithms', 'clean-code', 'mathematics',
}

# Keywords that identify implementation/solve tasks
IMPLEMENT_KEYWORDS: set[str] = {
    'implement', 'solve', 'write', 'build', 'create', 'code',
    'given', 'return the', 'return minimum', 'return maximum',
    'leetcode', 'algorithm', 'function that', 'program that',
    'find the', 'compute', 'calculate', 'design',
}

# Languages that use braces for code blocks
BRACE_LANGUAGES: set[str] = {'.go', '.c', '.cpp', '.rs', '.js', '.ts', '.java', '.cs', '.kt', '.swift'}
CHROMA_CHUNK_OVERLAP = 64               # optional for new ingest

# fitler settings for query pipeline
USE_SMART_K = True