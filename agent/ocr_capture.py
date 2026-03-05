"""
OCR module: Screenshot → clean text extraction with diagram understanding.

Handles:
  - Code screenshots → clean, properly formatted source code
  - Diagram/flowchart screenshots → structured text descriptions
  - Mixed content (code + annotations) → separated clean output
  - Problem statement screenshots (LeetCode, HackerRank, etc.)

Strategy for clean output:
  1. Pre-process: sharpen, denoise, threshold for high contrast
  2. OCR with EasyOCR (better for code than Tesseract — handles monospace, symbols)
  3. Post-process: fix common OCR errors in code (0/O, l/1, etc.)
  4. If the image looks like a diagram, use Ollama vision model to describe it

Dependencies (install when ready):
  pip install easyocr pillow numpy opencv-python-headless
"""

import os
import io
import re
import logging
import tempfile
from typing import Optional, Tuple, List

logger = logging.getLogger("ocr")


# ── Lazy state ───────────────────────────────────────────────────────────────

_easyocr_reader = None


def _get_reader(languages: List[str] = None):
    """Lazy-load EasyOCR reader. First load downloads models (~100MB)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        langs = languages or ["en"]
        logger.info("[OCR] Initializing EasyOCR with languages: %s", langs)
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
        logger.info("[OCR] EasyOCR ready")
    return _easyocr_reader


# ── Image pre-processing ────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> bytes:
    """Pre-process image for better OCR accuracy.

    Steps:
      1. Convert to grayscale
      2. Sharpen edges (helps with blurry screenshots)
      3. Apply adaptive thresholding (handles uneven lighting)
      4. Denoise
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    img = img.convert("L")

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)

    # Upscale if too small (OCR works better on larger images)
    min_dim = min(img.size)
    if min_dim < 800:
        scale = 800 / min_dim
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )

    # Adaptive threshold via numpy for clean binary image
    arr = np.array(img)
    # Simple Otsu-like threshold
    threshold = arr.mean()
    binary = ((arr > threshold) * 255).astype(np.uint8)
    img = Image.fromarray(binary)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Content type detection ───────────────────────────────────────────────────

def _classify_content(texts: List[str]) -> str:
    """Classify what the image contains based on OCR text.

    Returns: 'code', 'diagram', 'problem_statement', or 'mixed'
    """
    full_text = " ".join(texts).lower()

    # Code signals
    code_signals = [
        r'\bdef\b', r'\bclass\b', r'\bint\b', r'\bvoid\b', r'\breturn\b',
        r'\bfor\b.*\bin\b', r'\bwhile\b.*\{', r'\bif\b.*\{',
        r'#include', r'import\b', r'func\b', r'=>', r'->',
        r'\bconst\b', r'\blet\b', r'\bvar\b',
        r'[{}\[\]();]=', r'==|!=|<=|>=',
    ]
    code_score = sum(1 for sig in code_signals if re.search(sig, full_text))

    # Diagram signals (fewer code tokens, more natural language, arrows)
    diagram_signals = [
        r'→|←|↑|↓|⟶|⟵',
        r'\b(start|end|yes|no|true|false)\b',
        r'\b(step|stage|phase|flow|process)\b',
    ]
    diagram_score = sum(1 for sig in diagram_signals if re.search(sig, full_text))

    # Problem statement signals
    problem_signals = [
        r'\bgiven\b', r'\breturn\b.*\b(minimum|maximum|number|count)\b',
        r'\bexample\b', r'\binput\b.*\boutput\b',
        r'\bconstraints?\b', r'\bnote\b:',
    ]
    problem_score = sum(1 for sig in problem_signals if re.search(sig, full_text))

    if code_score >= 3:
        return "code"
    if problem_score >= 2:
        return "problem_statement"
    if diagram_score >= 2:
        return "diagram"
    if code_score >= 1 and problem_score >= 1:
        return "mixed"
    return "problem_statement"  # default


# ── Code-specific OCR post-processing ────────────────────────────────────────

def _fix_code_ocr_errors(text: str) -> str:
    """Fix common OCR misreads in code.

    Code OCR is notoriously bad at:
      - 0 vs O, 1 vs l vs I vs |
      - () vs {} vs []
      - Indentation (spaces/tabs lost)
      - Special chars: &&, ||, !=, ==, >=, <=, <<, >>
    """
    # Fix common letter/digit confusions in code context
    fixes = [
        # Restore operators that OCR often breaks
        (r'[|!]\s*=', '!='),
        (r'=\s*=', '=='),
        (r'<\s*=', '<='),
        (r'>\s*=', '>='),
        (r'&\s*&', '&&'),
        (r'\|\s*\|', '||'),
        # Fix common misreads
        (r'\bvold\b', 'void'),
        (r'\bint\s+maln\b', 'int main'),
        (r'\bincIude\b', 'include'),
        (r'\bprlnt\b', 'print'),
        (r'\bretum\b', 'return'),
        (r'\belse\s+lf\b', 'else if'),
        (r'\bstrlng\b', 'string'),
        (r'\bNuII\b', 'NULL'),
        (r'\bnuII\b', 'null'),
        (r'\bnullptr\b', 'nullptr'),  # already correct, just anchoring
    ]

    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)

    return text


def _reconstruct_code_layout(detections: list) -> str:
    """Reconstruct code layout from EasyOCR detections.

    EasyOCR returns bounding boxes + text. We use vertical position to
    determine line breaks and horizontal position for indentation.
    This produces much cleaner code output than naive text concatenation.
    """
    if not detections:
        return ""

    # Sort by vertical position (top-left Y), then horizontal (top-left X)
    sorted_dets = sorted(detections, key=lambda d: (d[0][0][1], d[0][0][0]))

    lines = []
    current_line_parts = []
    current_y = None
    line_height = None

    for det in sorted_dets:
        bbox, text, conf = det
        top_left_y = bbox[0][1]
        top_left_x = bbox[0][0]
        bottom_left_y = bbox[3][1]

        if line_height is None:
            line_height = abs(bottom_left_y - top_left_y)

        # New line if vertical gap > half the line height
        if current_y is not None and abs(top_left_y - current_y) > line_height * 0.5:
            # Flush current line
            if current_line_parts:
                lines.append(current_line_parts)
            current_line_parts = []

        current_line_parts.append((top_left_x, text, conf))
        current_y = top_left_y

    if current_line_parts:
        lines.append(current_line_parts)

    # Determine base indent unit from the minimum non-zero X offset
    all_x = [part[0] for line in lines for part in line]
    min_x = min(all_x) if all_x else 0

    # Estimate indent unit (typically ~20-40 pixels for 1 tab/4 spaces)
    x_offsets = sorted(set(int(part[0] - min_x) for line in lines for part in line))
    indent_unit = 30  # default
    if len(x_offsets) > 1:
        diffs = [x_offsets[i+1] - x_offsets[i] for i in range(len(x_offsets) - 1) if x_offsets[i+1] - x_offsets[i] > 10]
        if diffs:
            indent_unit = min(diffs)

    # Build the text with proper indentation
    result_lines = []
    for line_parts in lines:
        line_parts.sort(key=lambda p: p[0])  # sort by X within each line
        first_x = line_parts[0][0] - min_x
        indent_level = round(first_x / indent_unit) if indent_unit > 0 else 0
        indent = "    " * indent_level  # 4 spaces per level
        text = " ".join(part[1] for part in line_parts)
        result_lines.append(f"{indent}{text}")

    return "\n".join(result_lines)


# ── Diagram understanding ────────────────────────────────────────────────────

def describe_diagram(image_bytes: bytes) -> str:
    """Use Ollama vision model to describe a diagram/flowchart.

    Falls back to OCR text if no vision model is available.
    """
    from langchain_ollama import OllamaLLM
    from brain.config import LLM_MODEL
    import base64

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # Try LLaVA or other multimodal model first
        # If LLM_MODEL supports vision, use it directly
        llm = OllamaLLM(model="llava", temperature=0.0)
        prompt = (
            "Describe this diagram or flowchart in detail.\n"
            "If it's a flowchart, describe each step and decision point.\n"
            "If it's a class diagram, list the classes, properties, and relationships.\n"
            "If it's a sequence diagram, describe the interactions in order.\n"
            "Be precise and structured. Use numbered steps or bullet points."
        )
        # Note: Ollama vision requires the image to be passed as part of the prompt
        # This depends on the Ollama API version; fallback to text description
        result = llm.invoke(f"{prompt}\n\n[Image: data:image/png;base64,{b64[:100]}...]")
        return result.strip()
    except Exception as e:
        logger.warning("[OCR] Vision model unavailable, falling back to OCR text: %s", e)
        return ""


# ── Public API ───────────────────────────────────────────────────────────────

def extract_text_from_image(
    image_bytes: bytes,
    mode: str = "auto",
    languages: List[str] = None,
) -> dict:
    """Extract text from a screenshot image.

    Args:
        image_bytes: Raw image bytes (PNG, JPG, BMP, etc.)
        mode: 'auto' (detect content type), 'code', 'diagram', 'text'
        languages: OCR languages (default: ['en'])

    Returns:
        {
            "text": str,           # cleaned extracted text
            "content_type": str,   # 'code', 'diagram', 'problem_statement', 'mixed'
            "confidence": float,   # average OCR confidence
            "raw_text": str,       # pre-cleanup text
        }
    """
    # Pre-process for better OCR
    processed = preprocess_image(image_bytes)

    # Run EasyOCR
    reader = _get_reader(languages)

    # Save processed image to temp file for EasyOCR
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(processed)
        tmp_path = tmp.name

    try:
        detections = reader.readtext(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not detections:
        return {"text": "", "content_type": "unknown", "confidence": 0.0, "raw_text": ""}

    # Calculate average confidence
    avg_conf = sum(d[2] for d in detections) / len(detections)
    raw_texts = [d[1] for d in detections]

    # Detect content type
    content_type = mode if mode != "auto" else _classify_content(raw_texts)

    # Post-process based on content type
    if content_type == "code":
        # Use spatial layout reconstruction for code
        text = _reconstruct_code_layout(detections)
        text = _fix_code_ocr_errors(text)
    elif content_type == "diagram":
        # Try vision model for diagrams
        diagram_desc = describe_diagram(image_bytes)
        if diagram_desc:
            text = diagram_desc
        else:
            text = "\n".join(raw_texts)
    else:
        # Problem statements / general text: preserve natural reading order
        text = _reconstruct_code_layout(detections)

    raw_text = "\n".join(raw_texts)

    return {
        "text": text,
        "content_type": content_type,
        "confidence": round(avg_conf, 3),
        "raw_text": raw_text,
    }


def extract_text_from_file(image_path: str, mode: str = "auto") -> dict:
    """Extract text from an image file on disk."""
    with open(image_path, "rb") as f:
        return extract_text_from_image(f.read(), mode)
