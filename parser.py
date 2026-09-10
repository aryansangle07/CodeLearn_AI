import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeSymbol:
    name: str
    symbol_type: str  # "function" | "class" | "method" | "module"
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    parent_symbol: Optional[str] = None


@dataclass
class ParsedFile:
    file_path: str
    language: str
    total_lines: int
    symbols: List[CodeSymbol]
    raw_content: str


class PythonASTVisitor(ast.NodeVisitor):
    def __init__(self, total_lines: int):
        self.symbols: List[CodeSymbol] = []
        self.current_class: Optional[str] = None
        self.total_lines = total_lines

    def visit_Module(self, node: ast.Module) -> None:
        docstring = ast.get_docstring(node)
        if docstring:
            self.symbols.append(
                CodeSymbol(
                    name="<module>",
                    symbol_type="module",
                    start_line=1,
                    end_line=self.total_lines,
                    docstring=docstring.strip(),
                    parameters=[],
                    parent_symbol=None,
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        docstring = ast.get_docstring(node)
        end_lineno = getattr(node, "end_lineno", node.lineno)
        class_symbol = CodeSymbol(
            name=node.name,
            symbol_type="class",
            start_line=node.lineno,
            end_line=end_lineno,
            docstring=docstring.strip() if docstring else None,
            parameters=[getattr(base, "id", getattr(base, "attr", "Base")) for base in node.bases if hasattr(base, "id") or hasattr(base, "attr")],
            parent_symbol=self.current_class,
        )
        self.symbols.append(class_symbol)

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "anonymous")
        docstring = ast.get_docstring(node)
        end_lineno = getattr(node, "end_lineno", node.lineno)
        params = [arg.arg for arg in node.args.args]

        symbol_type = "method" if self.current_class else "function"
        parent = self.current_class

        self.symbols.append(
            CodeSymbol(
                name=name,
                symbol_type=symbol_type,
                start_line=node.lineno,
                end_line=end_lineno,
                docstring=docstring.strip() if docstring else None,
                parameters=params,
                parent_symbol=parent,
            )
        )
        # Note: Do not recurse into nested function bodies for sub-functions unless needed


class CodeParser:
    @classmethod
    def parse_python_ast(cls, file_path: str, content: str) -> ParsedFile:
        """
        Parses Python code into AST symbols, docstrings, and line spans.
        Falls back smoothly to heuristic parsing if syntax errors are encountered.
        """
        lines = content.splitlines()
        total_lines = len(lines) if lines else 1

        try:
            tree = ast.parse(content, filename=file_path)
            visitor = PythonASTVisitor(total_lines=total_lines)
            visitor.visit(tree)
            return ParsedFile(
                file_path=file_path,
                language="Python",
                total_lines=total_lines,
                symbols=visitor.symbols,
                raw_content=content,
            )
        except SyntaxError:
            # Fallback to heuristic parser
            return cls.parse_heuristic(file_path=file_path, content=content, language="Python")

    @classmethod
    def parse_heuristic(cls, file_path: str, content: str, language: str = "Other") -> ParsedFile:
        """
        Fallback parser for non-Python assets or malformed code using regex structural patterns.
        """
        lines = content.splitlines()
        total_lines = len(lines) if lines else 1
        symbols: List[CodeSymbol] = []

        if language == "Markdown":
            # Extract markdown section headings
            heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")
            headings = []
            for idx, line in enumerate(lines, start=1):
                m = heading_regex.match(line)
                if m:
                    headings.append((idx, m.group(2).strip()))

            for i, (start, title) in enumerate(headings):
                end = headings[i + 1][0] - 1 if i + 1 < len(headings) else total_lines
                symbols.append(
                    CodeSymbol(
                        name=title,
                        symbol_type="section",
                        start_line=start,
                        end_line=max(start, end),
                        docstring=None,
                        parameters=[],
                    )
                )

        elif language in ("JavaScript", "TypeScript"):
            # Heuristic JS/TS function & class regex
            func_regex = re.compile(
                r"(?:export\s+)?(?:async\s+)?(?:function\s+([a-zA-Z0-9_$]+)|class\s+([a-zA-Z0-9_$]+)|(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)"
            )
            for idx, line in enumerate(lines, start=1):
                m = func_regex.search(line)
                if m:
                    name = m.group(1) or m.group(2) or m.group(3)
                    sym_type = "class" if "class " in line else "function"
                    symbols.append(
                        CodeSymbol(
                            name=name,
                            symbol_type=sym_type,
                            start_line=idx,
                            end_line=min(idx + 30, total_lines),
                            docstring=None,
                            parameters=[],
                        )
                    )

        elif language == "Go":
            # Heuristic Go func/type regex
            go_regex = re.compile(r"func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\(")
            for idx, line in enumerate(lines, start=1):
                m = go_regex.search(line)
                if m:
                    symbols.append(
                        CodeSymbol(
                            name=m.group(1),
                            symbol_type="function",
                            start_line=idx,
                            end_line=min(idx + 25, total_lines),
                            docstring=None,
                            parameters=[],
                        )
                    )

        # If no symbols detected or other format, chunk by block units
        if not symbols:
            block_size = 50
            for start in range(1, total_lines + 1, block_size):
                end = min(start + block_size - 1, total_lines)
                symbols.append(
                    CodeSymbol(
                        name=f"block_{start}_{end}",
                        symbol_type="block",
                        start_line=start,
                        end_line=end,
                        docstring=None,
                        parameters=[],
                    )
                )

        return ParsedFile(
            file_path=file_path,
            language=language,
            total_lines=total_lines,
            symbols=symbols,
            raw_content=content,
        )

    @classmethod
    def parse_file(cls, file_path: str, content: str, language: str) -> ParsedFile:
        """Unified entrypoint dispatching to Python AST or Heuristic parser."""
        if language == "Python":
            return cls.parse_python_ast(file_path=file_path, content=content)
        return cls.parse_heuristic(file_path=file_path, content=content, language=language)
