import hashlib
from dataclasses import dataclass
from typing import List
from parser import ParsedFile, CodeSymbol


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    token_count: int
    language: str


class CodeChunker:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximates token count using character/word heuristics (1 token ≈ 4 chars)."""
        return max(1, len(text) // 4)

    @classmethod
    def _generate_chunk_id(cls, file_path: str, symbol_name: str, start_line: int, end_line: int) -> str:
        raw_key = f"{file_path}::{symbol_name}::{start_line}-{end_line}"
        hash_val = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
        safe_sym = symbol_name.replace("<", "").replace(">", "").strip()
        return f"{file_path}:{safe_sym}:{start_line}_{end_line}_{hash_val}"

    @classmethod
    def create_chunks(cls, parsed_file: ParsedFile, max_chunk_chars: int = 1500) -> List[CodeChunk]:
        """
        Transforms parsed symbols and file content into token-efficient semantic chunks
        preserving symbol hierarchies, line bounds, and language context.
        """
        lines = parsed_file.raw_content.splitlines(keepends=True)
        total_lines = len(lines)
        chunks: List[CodeChunk] = []

        if not lines:
            return chunks

        if not parsed_file.symbols:
            # Chunk entire file into sliding line blocks
            step_lines = 40
            for start in range(1, total_lines + 1, step_lines):
                end = min(start + step_lines - 1, total_lines)
                chunk_text = "".join(lines[start - 1 : end])
                chunk_id = cls._generate_chunk_id(parsed_file.file_path, "file_block", start, end)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed_file.file_path,
                        symbol_name="file_block",
                        symbol_type="block",
                        start_line=start,
                        end_line=end,
                        content=chunk_text,
                        token_count=cls.estimate_tokens(chunk_text),
                        language=parsed_file.language,
                    )
                )
            return chunks

        for sym in parsed_file.symbols:
            start_idx = max(0, sym.start_line - 1)
            end_idx = min(total_lines, sym.end_line)

            if start_idx >= end_idx:
                continue

            symbol_lines = lines[start_idx:end_idx]
            raw_chunk_text = "".join(symbol_lines)

            # Check if symbol text fits within max_chunk_chars
            if len(raw_chunk_text) <= max_chunk_chars:
                chunk_id = cls._generate_chunk_id(parsed_file.file_path, sym.name, sym.start_line, sym.end_line)
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=parsed_file.file_path,
                        symbol_name=sym.name,
                        symbol_type=sym.symbol_type,
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content=raw_chunk_text,
                        token_count=cls.estimate_tokens(raw_chunk_text),
                        language=parsed_file.language,
                    )
                )
            else:
                # Sub-chunk oversized symbols by line blocks
                sub_step = 30
                num_symbol_lines = len(symbol_lines)
                for offset in range(0, num_symbol_lines, sub_step):
                    sub_end_offset = min(offset + sub_step, num_symbol_lines)
                    sub_text = "".join(symbol_lines[offset:sub_end_offset])
                    actual_start = sym.start_line + offset
                    actual_end = sym.start_line + sub_end_offset - 1
                    chunk_id = cls._generate_chunk_id(
                        parsed_file.file_path, f"{sym.name}_part_{offset}", actual_start, actual_end
                    )
                    chunks.append(
                        CodeChunk(
                            chunk_id=chunk_id,
                            file_path=parsed_file.file_path,
                            symbol_name=f"{sym.name} (part {offset // sub_step + 1})",
                            symbol_type=sym.symbol_type,
                            start_line=actual_start,
                            end_line=actual_end,
                            content=sub_text,
                            token_count=cls.estimate_tokens(sub_text),
                            language=parsed_file.language,
                        )
                    )

        return chunks
