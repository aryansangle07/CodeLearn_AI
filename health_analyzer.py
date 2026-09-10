import os
import re
from typing import Dict, List, Tuple
from db import RunnabilityStatus
from schemas import HealthDetailsDTO

README_FILENAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}

DEPENDENCY_MANIFESTS = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "pipfile", "pipfile.lock", "environment.yml", "package.json",
    "go.mod", "cargo.toml", "pom.xml", "build.gradle", "gemfile", "composer.json"
}

ENTRY_POINT_FILENAMES = {
    "main.py", "app.py", "cli.py", "server.py", "manage.py",
    "wsgi.py", "asgi.py", "run.py", "index.js", "index.ts",
    "main.go", "main.rs", "application.java"
}

STUB_REGEX = re.compile(
    r"^\s*(?:pass|\.\.\.|raise\s+NotImplementedError(?:[^\n]*)|TODO\(\)|unimplemented!\(\))\s*$",
    re.MULTILINE,
)

TODO_REGEX = re.compile(
    r"(?:#|//|/\*|<!--)\s*(?:TODO|FIXME|XXX|HACK)\b",
    re.IGNORECASE,
)


class HealthAnalyzer:
    @classmethod
    def evaluate_repository(
        cls,
        safe_files: List[str],
        file_contents: Dict[str, str],
    ) -> Tuple[RunnabilityStatus, HealthDetailsDTO]:
        """
        Deterministically evaluates repository health and execution readiness:
        - Detects documentation (README)
        - Detects dependency manifests (requirements.txt, package.json, etc.)
        - Detects standard entrypoints (main.py, app.py, index.js, etc.)
        - Counts code stubs/empty implementations and TODO comments
        - Returns (RunnabilityStatus, HealthDetailsDTO)
        """
        detected_readmes: List[str] = []
        detected_manifests: List[str] = []
        detected_entrypoints: List[str] = []

        total_stubs = 0
        total_todos = 0

        for file_path in safe_files:
            base_name = os.path.basename(file_path).lower()

            if base_name in README_FILENAMES:
                detected_readmes.append(file_path)

            if base_name in DEPENDENCY_MANIFESTS:
                detected_manifests.append(file_path)

            if base_name in ENTRY_POINT_FILENAMES:
                detected_entrypoints.append(file_path)

            # Analyze file content if available
            content = file_contents.get(file_path, "")
            if content:
                # Check for main execution guard in Python files
                if file_path.endswith(".py") and file_path not in detected_entrypoints:
                    if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content:
                        detected_entrypoints.append(file_path)

                total_stubs += len(STUB_REGEX.findall(content))
                total_todos += len(TODO_REGEX.findall(content))

        has_readme = len(detected_readmes) > 0
        has_manifest = len(detected_manifests) > 0
        has_entrypoint = len(detected_entrypoints) > 0

        # Runnability evaluation hierarchy
        if has_readme and has_manifest and has_entrypoint:
            runnability = RunnabilityStatus.RUNNABLE
            summary = "Fully runnable repository with documented entrypoints, manifests, and documentation."
        elif (has_manifest and has_entrypoint) or (has_readme and (has_manifest or has_entrypoint)):
            runnability = RunnabilityStatus.PARTIALLY_RUNNABLE
            summary = "Partially runnable repository. Missing dedicated entrypoint or complete package manifest."
        else:
            runnability = RunnabilityStatus.ANALYSIS_ONLY
            summary = "Analysis-only repository. No active executable entrypoint or dependency manifests detected."

        health_dto = HealthDetailsDTO(
            has_readme=has_readme,
            has_dependency_manifest=has_manifest,
            dependency_files=detected_manifests,
            has_entry_point=has_entrypoint,
            entry_point_files=detected_entrypoints,
            empty_implementations_count=total_stubs,
            todo_comment_count=total_todos,
            summary_assessment=summary,
        )

        return runnability, health_dto
