import re
from typing import Dict, List, Optional

import datasets

# Import functions from the original utils using absolute import
from lm_eval.tasks.hendrycks_math.utils import (
    is_equiv,
    last_boxed_only_string,
    process_docs,
    remove_boxed,
)


def _extract_candidate_answer(text: str) -> Optional[str]:
    """Extract candidate answer from model output with multiple fallback strategies."""
    # Prefer \boxed / \fbox content.
    boxed = _last_boxed_only_string(text)
    if boxed is not None:
        unboxed = _remove_boxed(boxed)
        if unboxed is not None:
            return unboxed

    # Otherwise, take the last $...$ span.
    m = re.search(r"\$(.+?)\$(?!.*\$)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # As a last resort, try to grab the tail after "Answer:".
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()

    return None


def _last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{} or \\fbox{} string from the text."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> Optional[str]:
    """Remove \\boxed{} wrapper from string, with error handling."""
    try:
        if "\\boxed " in s:
            left = "\\boxed "
            if s[: len(left)] == left:
                return s[len(left) :]

        left = "\\boxed{"
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
    except Exception:
        pass
    return None


def process_docs_test(dataset: datasets.Dataset) -> datasets.Dataset:
    """For math500_reasoning_test: only keep the first 10 samples."""
    return dataset.select(list(range(min(10, len(dataset))))) if len(dataset) > 0 else dataset


def process_results_boxed(doc: dict, results: List[str]) -> Dict[str, int]:
    """Process results for boxed answer format - extracts answer from \\boxed{} in model output."""
    retval = 0
    model_output = results[0] if results else ""
    
    # Extract candidate answer using improved extraction with fallbacks
    answer = _extract_candidate_answer(model_output)
    if answer is None:
        # Final fallback: use the entire output
        answer = model_output.strip()

    # Compare with gold answer (extracted from solution)
    gold_answer = remove_boxed(last_boxed_only_string(doc["solution"]))
    
    if is_equiv(answer, gold_answer):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results

