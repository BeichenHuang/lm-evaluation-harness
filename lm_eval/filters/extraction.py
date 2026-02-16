import re
import sys
import unicodedata

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def _last_boxed_content(string: str) -> str | None:
    """Extract content from the last \\boxed{...} in a string, handling nested braces."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    # find the opening brace
    i = idx + len("\\boxed")
    if i >= len(string) or string[i] != "{":
        return None
    # match braces
    depth = 0
    start = i
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[start + 1 : i]
        i += 1
    return None


def _clean_latex_number(s: str) -> str:
    """Clean LaTeX-formatted number: remove {,} separators, $, spaces, \\text{}, etc."""
    s = s.replace("{,}", "")
    s = s.replace("\\,", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.replace("$", "")
    s = s.replace("\\$", "")
    s = s.replace("%", "")
    s = s.replace("\\%", "")
    s = s.rstrip(".")
    return s.strip()


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps


@register_filter("regex_pos")
class POSFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags if pos_tags else self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        filtered_resps = map(lambda x: filter_set(x), resps)

        return filtered_resps


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading and trailing whitespace from responses."""

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp = resp.strip()
                filtered_resp.append(resp)
            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex: r's*([A-?])', where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            if not isinstance(resp, str):
                resp = ""
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


@register_filter("thinking_model_extract")
class ThinkingModelExtractFilter(Filter):
    """A filter for thinking models (e.g., Qwen3-Thinking) that produce
    <think>...</think> blocks and may output answers in \\boxed{} or #### format.

    Extraction priority:
      1. \\boxed{NUMBER} (with LaTeX cleanup)
      2. #### NUMBER
      3. Last number in the text (flexible fallback)
    """

    def __init__(
        self,
        strip_think: bool = True,
        fallback: str = "[invalid]",
    ) -> None:
        self.strip_think = strip_think
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def extract_answer(resp: str) -> str:
            if not isinstance(resp, str) or not resp:
                return self.fallback

            text = resp
            # Strip <think>...</think> content so we only look at the final answer
            if self.strip_think and "</think>" in text:
                text = text.split("</think>")[-1]

            # 1) Try \\boxed{...}
            boxed = _last_boxed_content(text)
            if boxed is not None:
                cleaned = _clean_latex_number(boxed)
                if cleaned:
                    return cleaned

            # 2) Try #### NUMBER
            m = re.search(r"####\s*(\-?[0-9\.\,]+)", text)
            if m:
                return m.group(1).replace(",", "").strip()

            # 3) Fallback: last bare number in text
            nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
            if nums:
                return nums[-1].replace(",", "")

            return self.fallback

        def filter_set(inst: list[str]) -> list[str]:
            return [extract_answer(r) for r in inst]

        return [filter_set(r) for r in resps]
