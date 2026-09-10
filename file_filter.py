import os
from typing import Dict, List, Tuple
from config import get_settings

BLOCKED_EXTENSIONS = {
    ".env", ".pem", ".key", ".p12", ".crt", ".cert", ".der", ".pfx",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".iso", ".img",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".parquet", ".arrow",
    ".db", ".sqlite", ".sqlite3", ".pyc", ".pyo", ".pyd",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp4", ".mp3", ".wav", ".pdf", ".docx", ".xlsx", ".woff", ".woff2", ".ttf", ".eot"
}

BLOCKED_DIRECTORIES = {
    ".git", ".github", "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".idea", ".vscode", "dist", "build",
    ".mypy_cache", ".ruff_cache", "target", "coverage", ".next", ".nuxt"
}

EXTENSION_TO_LANGUAGE = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++",
    ".hpp": "C++",
    ".rs": "Rust",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Shell",
    ".html": "HTML",
    ".css": "CSS",
    ".txt": "Text",
}


class FileFilter:
    @classmethod
    def is_binary_file(cls, file_path: str) -> bool:
        """Heuristic check for binary files by scanning the first 1024 bytes for null bytes."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
            return False
        except Exception:
            return True

    @classmethod
    def is_safe_file(
        cls,
        relative_path: str,
        file_size_bytes: int,
        max_file_size_mb: float = 2.0
    ) -> bool:
        """
        Enforces extension blacklist, directory blacklist, and file size limits.
        """
        # Normalize slashes
        norm_path = relative_path.replace("\\", "/")
        parts = norm_path.strip("/").split("/")

        # Check directory components
        for part in parts[:-1]:
            if part in BLOCKED_DIRECTORIES or part.startswith("."):
                return False

        filename = parts[-1]
        if not filename:
            return False

        # Block hidden files / dotfiles (e.g., .env, .env.local, .DS_Store)
        if filename.startswith("."):
            return False

        # Check extension
        _, ext = os.path.splitext(filename.lower())
        if ext in BLOCKED_EXTENSIONS:
            return False

        # Check file size limit
        max_bytes = max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            return False

        return True

    @classmethod
    def get_file_language(cls, file_path: str) -> str:
        """Resolves programming/markup language from file extension."""
        _, ext = os.path.splitext(file_path.lower())
        return EXTENSION_TO_LANGUAGE.get(ext, "Other")

    @classmethod
    def scan_repository_tree(
        cls,
        root_dir: str,
        max_repo_size_mb: float = 50.0,
        max_files: int = 500
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Walks repository tree, filters out unsafe files, enforces boundary limits,
        and computes percentage language distribution based on file counts.
        """
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            raise ValueError(f"Repository directory does not exist: {root_dir}")

        safe_files: List[str] = []
        total_size_bytes = 0
        language_counts: Dict[str, int] = {}
        max_repo_bytes = max_repo_size_mb * 1024 * 1024

        for root, dirs, files in os.walk(root_dir):
            # In-place modify dirs to avoid traversing blocked directories
            dirs[:] = [d for d in dirs if d not in BLOCKED_DIRECTORIES and not d.startswith(".")]

            for file in sorted(files):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")

                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    continue

                if not cls.is_safe_file(rel_path, file_size):
                    continue

                # Exclude binary files
                if cls.is_binary_file(full_path):
                    continue

                total_size_bytes += file_size
                if total_size_bytes > max_repo_bytes:
                    raise ValueError(
                        f"Repository exceeds maximum allowed size of {max_repo_size_mb} MB."
                    )

                safe_files.append(rel_path)

                if len(safe_files) > max_files:
                    raise ValueError(
                        f"Repository exceeds maximum allowed file count limit of {max_files} files."
                    )

                lang = cls.get_file_language(rel_path)
                language_counts[lang] = language_counts.get(lang, 0) + 1

        # Calculate percentage distribution
        total_counted = len(safe_files)
        language_distribution: Dict[str, float] = {}
        if total_counted > 0:
            for lang, count in language_counts.items():
                percentage = round((count / total_counted) * 100.0, 1)
                language_distribution[lang] = percentage

        return safe_files, language_distribution
