"""
Tests for the indexing pipeline.

Run with: pytest tests/test_indexing.py -v
"""

import os
import json
import pytest
from pathlib import Path


class TestChunksJsonl:
    """Tests for the chunks.jsonl output file."""
    
    @pytest.fixture
    def chunks_path(self):
        """Path to chunks.jsonl file."""
        return Path(__file__).parent.parent / "out" / "chunks.jsonl"
    
    def test_chunks_file_exists(self, chunks_path):
        """Test that chunks.jsonl exists."""
        assert chunks_path.exists(), "chunks.jsonl file should exist"
    
    def test_chunks_are_valid_json(self, chunks_path):
        """Test that each line is valid JSON."""
        if not chunks_path.exists():
            pytest.skip("chunks.jsonl not found")
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON on line {i+1}: {e}")
    
    def test_chunks_have_required_fields(self, chunks_path):
        """Test that chunks have all required fields."""
        if not chunks_path.exists():
            pytest.skip("chunks.jsonl not found")
        
        required_fields = {"id", "text", "work", "work_abbrev", "location", "section_type", "title"}
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    missing = required_fields - set(chunk.keys())
                    if missing:
                        pytest.fail(f"Chunk on line {i+1} missing fields: {missing}")
                    if i >= 100:  # Only check first 100 for speed
                        break
    
    def test_chunks_have_non_empty_text(self, chunks_path):
        """Test that chunks have non-empty text."""
        if not chunks_path.exists():
            pytest.skip("chunks.jsonl not found")
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    assert chunk.get("text"), f"Chunk on line {i+1} has empty text"
                    if i >= 100:
                        break
    
    def test_work_abbrevs_are_valid(self, chunks_path):
        """Test that work abbreviations are from the expected set."""
        if not chunks_path.exists():
            pytest.skip("chunks.jsonl not found")
        
        valid_abbrevs = {"ST", "SCG", "DV", "DPD", "QDA", "DBE", "CM"}
        found_abbrevs = set()
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    abbrev = chunk.get("work_abbrev", "")
                    found_abbrevs.add(abbrev)
                    if abbrev and abbrev not in valid_abbrevs:
                        pytest.fail(f"Invalid work_abbrev '{abbrev}' on line {i+1}")
        
        # Check that we have chunks from multiple works
        assert len(found_abbrevs) > 1, "Should have chunks from multiple works"


class TestIndexScript:
    """Tests for the index_to_qdrant.py script."""
    
    @pytest.fixture
    def script_path(self):
        """Path to indexing script."""
        return Path(__file__).parent.parent / "scripts" / "index_to_qdrant.py"
    
    def test_script_exists(self, script_path):
        """Test that indexing script exists."""
        assert script_path.exists(), "index_to_qdrant.py should exist"
    
    def test_script_is_valid_python(self, script_path):
        """Test that script is valid Python."""
        if not script_path.exists():
            pytest.skip("index_to_qdrant.py not found")
        
        import ast
        
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Invalid Python syntax: {e}")


class TestParseCorpusScript:
    """Tests for the parse_corpus.py script."""
    
    @pytest.fixture
    def script_path(self):
        """Path to parsing script."""
        return Path(__file__).parent.parent / "scripts" / "parse_corpus.py"
    
    def test_script_exists(self, script_path):
        """Test that parsing script exists."""
        assert script_path.exists(), "parse_corpus.py should exist"
    
    def test_script_is_valid_python(self, script_path):
        """Test that script is valid Python."""
        if not script_path.exists():
            pytest.skip("parse_corpus.py not found")
        
        import ast
        
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Invalid Python syntax: {e}")
