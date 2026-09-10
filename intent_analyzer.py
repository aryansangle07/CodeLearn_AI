import re
from schemas import IntentTypeEnum


class IntentAnalyzer:
    ARCHITECTURE_PATTERNS = [
        re.compile(r"\b(?:architecture|overview|high[- ]level|structure|topology|components|system\s+design|folder\s+structure|organization)\b", re.IGNORECASE),
        re.compile(r"\b(?:how\s+is\s+(?:the\s+)?(?:project|repo|codebase|system)\s+(?:structured|organized|designed))\b", re.IGNORECASE),
    ]

    SIMPLE_LOOKUP_PATTERNS = [
        re.compile(r"^\s*(?:where\s+is|find|locate|definition\s+of|which\s+file\s+(?:has|contains|defines))\b", re.IGNORECASE),
        re.compile(r"^\s*[a-zA-Z0-9_]{3,60}\s*\??\s*$"),  # Single symbol query (e.g. "parse_github_url?")
        re.compile(r"\b(?:where\s+(?:is|can\s+I\s+find)\s+(?:the\s+)?[a-zA-Z0-9_.]+)\b", re.IGNORECASE),
    ]

    # Explicit Out-of-Domain Blocklist Patterns
    OUT_OF_DOMAIN_PATTERNS = [
        r"\b(capital of|population of|president of|prime minister|weather in|forecast|temperature|recipe for|bake|cookie|cake|chocolate|who won|football|soccer|movies|movie|song|lyrics|translate to french|translate to spanish)\b",
        r"\b(what is the meaning of life|who is the ceo of|how to make money|bitcoin price|stock price)\b",
    ]

    # Explicit In-Scope Code/Technical Patterns
    CODE_TECHNICAL_PATTERNS = [
        r"\b(\.py|\.toml|\.json|\.md|\.yaml|\.yml|\.ts|\.js|\.rs|\.go|\.c|\.cpp)\b",
        r"\b(def |class |import |@|decorator|parameter|argument|parser|command|cli|context|function|method|route|middleware|endpoint|handler|callback|async|await|typing|type|module|subcommand)\b",
        r"\b(install|setup|build|dependency|requirements|pyproject|wheel|test|pytest|testsuite|license|author|version|git|repo|commit)\b",
    ]

    @classmethod
    def is_out_of_domain(cls, query: str) -> bool:
        """Deterministically checks if query matches explicit out-of-domain non-software patterns."""
        q_lower = query.lower()
        if any(re.search(pat, q_lower) for pat in cls.OUT_OF_DOMAIN_PATTERNS):
            # If it explicitly contains code extensions or technical keywords, don't reject
            if not any(re.search(pat, q_lower) for pat in cls.CODE_TECHNICAL_PATTERNS):
                return True
        return False

    @classmethod
    def classify_intent(cls, query: str) -> IntentTypeEnum:
        """
        Classifies user query intent deterministically into IntentTypeEnum.
        """
        if not query:
            return IntentTypeEnum.EXPLANATION

        q_lower = query.lower()

        # 1. Check for explicit out-of-domain queries
        if cls.is_out_of_domain(q_lower):
            return IntentTypeEnum.UNRELATED

        # 2. Check for comparative / grey-zone queries
        if any(w in q_lower for w in ["compare", "difference between", "versus", "vs", "better than", "alternative to"]):
            return IntentTypeEnum.COMPARATIVE

        # 3. Standard deterministic intent matching
        for pat in cls.ARCHITECTURE_PATTERNS:
            if pat.search(q_lower):
                return IntentTypeEnum.ARCHITECTURE
        for pat in cls.SIMPLE_LOOKUP_PATTERNS:
            if pat.search(q_lower):
                return IntentTypeEnum.SIMPLE_LOOKUP

        return IntentTypeEnum.EXPLANATION
