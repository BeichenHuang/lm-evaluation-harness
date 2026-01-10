import re
from importlib.metadata import version
from typing import Any, Dict, List, Optional

import datasets


try:
    import sympy
    from sympy import N
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy>=1.12` and `antlr4-python3-runtime==4.11` are required for BRUMO/HMMT grading. "
        "Please install via `pip install lm-eval[math]` or `pip install -e .[math]`."
    ) from e


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # No filtering needed; keep schema stable.
    return dataset


def process_docs_test(dataset: datasets.Dataset) -> datasets.Dataset:
    # For HMMT_test: only keep the first sample
    return dataset.select([1]) if len(dataset) > 0 else dataset


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    raw = results[0] if results else ""
    gold = str(doc.get("answer", ""))
    gold_is_list = "," in gold

    cand_str = _extract_candidate_answer(raw, list_answer=gold_is_list) or raw
    cand = _parse_answer(cand_str, list_answer=gold_is_list)
    gold_parsed = _parse_answer(gold, list_answer=gold_is_list)

    return {"exact_match": int(_check_answers(cand, gold_parsed))}


def _extract_candidate_answer(text: str, *, list_answer: bool = False) -> Optional[str]:
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
    try:
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _split_top_level_commas(s: str) -> List[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    opening = set(pairs.keys())
    closing = set(pairs.values())

    for ch in s:
        if ch in opening:
            depth += 1
        elif ch in closing:
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _latex2sympy_fixed(latex: str):
    # If _integer is present, replace it with _{integer} for any integer.
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    parsed = parse_latex(latex)
    known_constants = {"pi": sympy.pi, "e": sympy.E, "I": sympy.I, "i": sympy.I}
    return parsed.xreplace(
        {s: known_constants[s.name] for s in getattr(parsed, "free_symbols", set()) if s.name in known_constants}
    )


def _parse_single(expr: str):
    expr = expr.strip()
    if not expr:
        return None

    # Drop trivial wrappers/punctuation.
    expr = expr.strip().rstrip(".")
    expr = expr.replace("\\left", "").replace("\\right", "")

    # Prefer true LaTeX parsing if it looks like LaTeX.
    if "\\" in expr:
        try:
            return _latex2sympy_fixed(expr)
        except Exception:
            pass

    # SymPy-style parsing (also works for many cleaned LaTeX-ish strings).
    expr = expr.replace("^", "**")
    locals_map = {
        "sqrt": sympy.sqrt,
        "exp": sympy.exp,
        "pi": sympy.pi,
        "E": sympy.E,
        "I": sympy.I,
        "i": sympy.I,
    }
    try:
        return sympy.sympify(expr, locals=locals_map)
    except Exception:
        return expr


def _parse_answer(expr: str, *, list_answer: bool = False):
    expr = (expr or "").strip()
    if not expr:
        return None

    expr = expr.strip()
    if list_answer:
        # Normalize obvious list wrappers.
        if (expr.startswith("[") and expr.endswith("]")) or (expr.startswith("(") and expr.endswith(")")):
            expr = expr[1:-1].strip()
        items = _split_top_level_commas(expr)
        return [_parse_single(x) for x in items]
    return _parse_single(expr)


def _check_answers(ans1: Any, ans2: Any) -> bool:
    if ans1 is None or ans2 is None:
        return False

    is_list1 = isinstance(ans1, (list, tuple))
    is_list2 = isinstance(ans2, (list, tuple))
    if is_list1 != is_list2:
        return False

    if is_list1 and is_list2:
        if len(ans1) != len(ans2):
            return False
        used = set()
        for a in ans1:
            found = False
            for i, b in enumerate(ans2):
                if i in used:
                    continue
                if _check_answers(a, b):
                    used.add(i)
                    found = True
                    break
            if not found:
                return False
        return True

    # Prefer sympy's symbolic equality where possible.
    try:
        if not (hasattr(ans1, "equals") and callable(ans1.equals)) or not (
            hasattr(ans2, "equals") and callable(ans2.equals)
        ):
            if isinstance(ans1, str) or isinstance(ans2, str):
                return bool(str(ans1).strip() == str(ans2).strip())
            err = abs(N(ans1 - ans2))
            denom = max(abs(N(ans1)), abs(N(ans2)))
            return bool(err < 1e-10 and (denom == 0 or err / denom < 1e-10))
        return bool(ans1.equals(ans2))
    except Exception:
        return bool(str(ans1).strip() == str(ans2).strip())


