import os
import re
from typing import Any, Dict, List, Set, Tuple


class CitationValidator:
    # Matches backticked file references with standard extensions
    FILE_PATH_REGEX = re.compile(
        r"`([a-zA-Z0-9_.\-\/\\]+\.(?:py|js|ts|jsx|tsx|go|java|c|cpp|h|hpp|rs|md|json|yaml|yml|toml|sql|sh|txt))(?::(?:L|line\s*)?\d+(?:-(?:L|line\s*)?\d+)?)?`",
        re.IGNORECASE,
    )

    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """Extracts candidate file paths referenced in backticks from text."""
        if not text:
            return []

        matches = cls.FILE_PATH_REGEX.findall(text)
        cleaned_paths = []
        for m in matches:
            norm_path = m.replace("\\", "/").strip().lstrip("./")
            if norm_path and norm_path not in cleaned_paths:
                cleaned_paths.append(norm_path)

        return cleaned_paths

    @classmethod
    def validate_citations(
        cls, answer: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], bool]:
        """
        Validates extracted file citations against the retrieved chunk context.
        Returns:
            - validated_answer: Sanitized or adjusted answer string
            - verified_citations: List of verified file path strings
            - is_grounded: Boolean indicating if all claims are grounded in context
        """
        if not answer:
            return "The provided repository context is insufficient to answer this question accurately.", [], False

        valid_file_paths: Set[str] = {
            c.get("file_path", "").replace("\\", "/").strip().lstrip("./")
            for c in retrieved_chunks
            if c.get("file_path")
        }

        extracted_citations = cls.extract_citations(answer)
        verified_citations: List[str] = []
        hallucinated_citations: List[str] = []

        for cit in extracted_citations:
            # Check if extracted citation matches any retrieved file path or basename
            matched = False
            for v_path in valid_file_paths:
                if cit == v_path or v_path.endswith("/" + cit) or cit.endswith("/" + v_path):
                    if v_path not in verified_citations:
                        verified_citations.append(v_path)
                    matched = True
                    break
            if not matched:
                hallucinated_citations.append(cit)

        # If answer has hallucinated citations
        if hallucinated_citations:
            is_grounded = False
            # If no citations were verified, fallback to insufficient context
            if not verified_citations:
                final_answer = "The provided repository context is insufficient to answer this question accurately."
                return final_answer, [], False
            else:
                final_answer = answer + f"\n\n*(Note: Some referenced paths `{hallucinated_citations}` could not be verified in the retrieved code index.)*"
                return final_answer, verified_citations, False

        # If no citations were explicitly extracted from answer, but chunks exist
        if not verified_citations and valid_file_paths:
            # Check if answer mentions insufficient context
            if "insufficient to answer" in answer.lower():
                return answer, [], True
            # Include top retrieved chunk files as default verified references
            verified_citations = list(valid_file_paths)[:3]

        return answer, verified_citations, True
