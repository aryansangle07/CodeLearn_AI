import pytest
from db import RunnabilityStatus
from parser import CodeParser, ParsedFile
from chunker import CodeChunker
from health_analyzer import HealthAnalyzer


def test_python_ast_symbol_extraction():
    python_code = '''"""Module docstring for test module."""

class Calculator:
    """A sample calculator class."""
    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, x: int) -> int:
        """Adds x to current value."""
        self.value += x
        return self.value

def helper_function(msg: str) -> None:
    """Helper print utility."""
    print(f"Log: {msg}")
'''

    parsed = CodeParser.parse_python_ast("math/calc.py", python_code)
    assert parsed.language == "Python"
    assert parsed.total_lines > 10

    symbols_by_name = {s.name: s for s in parsed.symbols}
    assert "<module>" in symbols_by_name
    assert symbols_by_name["<module>"].docstring == "Module docstring for test module."

    assert "Calculator" in symbols_by_name
    assert symbols_by_name["Calculator"].symbol_type == "class"
    assert symbols_by_name["Calculator"].docstring == "A sample calculator class."

    assert "__init__" in symbols_by_name
    assert symbols_by_name["__init__"].symbol_type == "method"
    assert symbols_by_name["__init__"].parent_symbol == "Calculator"

    assert "add" in symbols_by_name
    assert symbols_by_name["add"].symbol_type == "method"
    assert symbols_by_name["add"].parent_symbol == "Calculator"

    assert "helper_function" in symbols_by_name
    assert symbols_by_name["helper_function"].symbol_type == "function"
    assert symbols_by_name["helper_function"].parent_symbol is None
    assert symbols_by_name["helper_function"].parameters == ["msg"]


def test_python_ast_syntax_error_fallback():
    broken_code = """
    def broken_func(
        # missing closing parenthesis and body
    class Foo:
    """

    parsed = CodeParser.parse_python_ast("broken.py", broken_code)
    assert parsed.language == "Python"
    assert len(parsed.symbols) > 0  # Heuristic chunking fallback occurred
    assert parsed.symbols[0].symbol_type == "block"


def test_heuristic_parser_non_python():
    # Markdown
    md_content = """# Main Title
Introductory text.

## Setup Instructions
Run pip install.

## Usage Guide
Run python main.py.
"""
    parsed_md = CodeParser.parse_heuristic("README.md", md_content, language="Markdown")
    assert parsed_md.language == "Markdown"
    md_symbol_names = [s.name for s in parsed_md.symbols]
    assert "Main Title" in md_symbol_names
    assert "Setup Instructions" in md_symbol_names
    assert "Usage Guide" in md_symbol_names

    # JavaScript
    js_content = """
    function calculateTotal(items) {
        return items.reduce((acc, x) => acc + x.price, 0);
    }

    class OrderService {
        processOrder(order) {
            return true;
        }
    }
    """
    parsed_js = CodeParser.parse_heuristic("app.js", js_content, language="JavaScript")
    assert parsed_js.language == "JavaScript"
    js_names = [s.name for s in parsed_js.symbols]
    assert "calculateTotal" in js_names
    assert "OrderService" in js_names


def test_code_chunker_semantic_boundaries():
    python_code = """class UserService:
    def get_user(self, user_id: str):
        return {"id": user_id, "name": "Alice"}

    def delete_user(self, user_id: str):
        return True
"""
    parsed = CodeParser.parse_python_ast("services/user.py", python_code)
    chunks = CodeChunker.create_chunks(parsed, max_chunk_chars=500)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.file_path == "services/user.py"
        assert chunk.token_count > 0
        assert chunk.start_line <= chunk.end_line
        assert chunk.language == "Python"
        assert len(chunk.content) > 0


def test_health_analyzer_runnable_repo():
    files = ["README.md", "requirements.txt", "main.py", "app/utils.py"]
    contents = {
        "README.md": "# Project\nRun with python main.py",
        "requirements.txt": "fastapi==0.111.0\nuvicorn==0.29.0",
        "main.py": "import uvicorn\nif __name__ == '__main__':\n    uvicorn.run('app:app')",
        "app/utils.py": "def helper():\n    return 42",
    }

    status, health_dto = HealthAnalyzer.evaluate_repository(files, contents)
    assert status == RunnabilityStatus.RUNNABLE
    assert health_dto.has_readme is True
    assert health_dto.has_dependency_manifest is True
    assert health_dto.has_entry_point is True
    assert "main.py" in health_dto.entry_point_files


def test_health_analyzer_partial_repo():
    files = ["README.md", "pyproject.toml", "src/core.py"]
    contents = {
        "README.md": "# Library\nInstall and import",
        "pyproject.toml": "[tool.poetry]\nname = 'mylib'",
        "src/core.py": "class CoreEngine:\n    pass",
    }

    status, health_dto = HealthAnalyzer.evaluate_repository(files, contents)
    assert status == RunnabilityStatus.PARTIALLY_RUNNABLE
    assert health_dto.has_readme is True
    assert health_dto.has_dependency_manifest is True
    assert health_dto.has_entry_point is False


def test_health_analyzer_analysis_only():
    files = ["scratch.py", "notes.txt"]
    contents = {
        "scratch.py": "def foo():\n    pass\n    # TODO: implement",
        "notes.txt": "just some notes",
    }

    status, health_dto = HealthAnalyzer.evaluate_repository(files, contents)
    assert status == RunnabilityStatus.ANALYSIS_ONLY
    assert health_dto.has_readme is False
    assert health_dto.has_dependency_manifest is False
    assert health_dto.has_entry_point is False
    assert health_dto.empty_implementations_count >= 1
    assert health_dto.todo_comment_count >= 1
