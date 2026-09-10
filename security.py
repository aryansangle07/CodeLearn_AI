import re
from typing import Tuple

SECRET_PATTERNS = [
    # AWS Access Key ID
    re.compile(r"\b((?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16})\b"),
    # AWS Secret Key (heuristic when following AWS key or keyword)
    re.compile(r"(?i)(?:aws_secret_access_key|aws_secret_key)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
    # JWT Tokens (header.payload.signature)
    re.compile(r"\b(eyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,})\b"),
    # RSA / EC / DSA / OpenSSH / Generic Private Key Blocks
    re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"),
    # GitHub Personal Access / OAuth Tokens
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,255})\b"),
    # Slack Tokens
    re.compile(r"\b(xox[baprs]-[0-9a-zA-Z]{10,48})\b"),
    # Generic API Keys / Passwords in configuration / code assignments
    re.compile(
        r"(?i)(?:api[_-]?key|client[_-]?secret|password|passwd|auth[_-]?token|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]([^'\"\r\n\s]{8,})['\"]"
    ),
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)<\s*/?\s*(?:system|inst|sys|prompt|context|assistant)\s*>"),
    re.compile(r"(?i)\[\s*/?\s*(?:INST|SYS)\s*\]"),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"),
    re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:DAN|developer|unrestricted|jailbreak)\s+mode\b"),
]


class SecurityScanner:
    @classmethod
    def redact_secrets(cls, content: str) -> Tuple[str, int]:
        """
        Scans content with regex patterns for sensitive credentials, keys, and tokens.
        Replaces matched secrets with '[REDACTED_SECRET]' and returns the sanitized content
        along with the total count of redactions performed.
        """
        if not content:
            return content, 0

        redacted_text = content
        total_redactions = 0

        # Redact private key blocks first
        private_key_pattern = SECRET_PATTERNS[3]
        matches = private_key_pattern.findall(redacted_text)
        if matches:
            total_redactions += len(matches)
            redacted_text = private_key_pattern.sub("[REDACTED_SECRET]", redacted_text)

        # Redact remaining patterns
        for pattern in SECRET_PATTERNS:
            if pattern == private_key_pattern:
                continue

            def _replace_match(match: re.Match) -> str:
                nonlocal total_redactions
                total_redactions += 1
                # If pattern has groups, preserve assignment key and redact value
                if match.lastindex and match.lastindex >= 1:
                    full_str = match.group(0)
                    secret_val = match.group(match.lastindex)
                    return full_str.replace(secret_val, "[REDACTED_SECRET]")
                return "[REDACTED_SECRET]"

            redacted_text = pattern.sub(_replace_match, redacted_text)

        return redacted_text, total_redactions

    @classmethod
    def sanitize_untrusted_prompt(cls, text: str) -> str:
        """
        Strips malicious prompt injection delimiters, control markers, and jailbreak phrases
        from untrusted repository content or user queries.
        """
        if not text:
            return ""

        sanitized = text
        for pattern in PROMPT_INJECTION_PATTERNS:
            sanitized = pattern.sub("[FILTERED_PROMPT_INJECTION]", sanitized)

        # Strip control characters (excluding newline, tab, carriage return)
        sanitized = "".join(ch for ch in sanitized if ch in "\n\r\t" or (ord(ch) >= 32 and ord(ch) != 127))

        return sanitized
