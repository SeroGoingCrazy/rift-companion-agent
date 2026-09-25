"""Tests for E2 – MarkdownLoader, LoaderFactory and section-aware chunking.

Covers:
- Heading-hierarchy section splitting (title_path, ancestor headings, fences)
- Document metadata: source, collection, title
- Extension-based loader selection through LoaderFactory
- DocumentChunker keeping chunks inside sections
- Pipeline SHA256 incremental skip for unchanged Markdown files
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.types import Chunk
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.pipeline import IngestionPipeline
from src.libs.loader import (
    BaseLoader,
    ExtensionRoutingLoader,
    LoaderFactory,
    MarkdownLoader,
    PdfLoader,
    SQLiteIntegrityChecker,
)
from src.libs.loader.markdown_loader import split_sections
from src.libs.splitter.base_splitter import BaseSplitter


REFUND_MD = """\
# 取消与退款

所有退款按原路退回。

## 退款档位

### 开局前 24 小时以上

全额退款。

### 开局前 3–24 小时

退款 50%。

## 陪玩师原因

陪玩师迟到或违规导致取消，全额退款。
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def refund_doc(tmp_path):
    return _write(tmp_path / "kb" / "platform_rules" / "refund.md", REFUND_MD)


def _titles(sections):
    return [s["title_path"] for s in sections]


# ── Section splitting ────────────────────────────────────────────────


class TestSplitSections:

    def test_multi_level_headings_produce_title_paths(self, refund_doc):
        doc = MarkdownLoader().load(refund_doc)

        assert _titles(doc.metadata["sections"]) == [
            "取消与退款",
            "取消与退款 > 退款档位 > 开局前 24 小时以上",
            "取消与退款 > 退款档位 > 开局前 3–24 小时",
            "取消与退款 > 陪玩师原因",
        ]

    def test_section_text_starts_with_ancestor_headings(self, refund_doc):
        sections = MarkdownLoader().load(refund_doc).metadata["sections"]

        assert sections[2]["text"] == (
            "# 取消与退款\n## 退款档位\n### 开局前 3–24 小时\n\n退款 50%。"
        )
        assert sections[2]["level"] == 3

    def test_heading_without_body_is_not_a_section(self, refund_doc):
        sections = MarkdownLoader().load(refund_doc).metadata["sections"]

        assert "取消与退款 > 退款档位" not in _titles(sections)

    def test_sibling_heading_pops_deeper_levels(self, refund_doc):
        sections = MarkdownLoader().load(refund_doc).metadata["sections"]

        # "陪玩师原因" is an H2 after H3s: the H3 path must not leak into it
        assert sections[-1]["text"].startswith("# 取消与退款\n## 陪玩师原因\n\n")

    def test_offsets_cover_the_section_in_the_original_text(self, refund_doc):
        doc = MarkdownLoader().load(refund_doc)
        section = doc.metadata["sections"][3]

        original = doc.text[section["start_offset"]:section["end_offset"]]
        assert original.startswith("## 陪玩师原因")
        assert "全额退款" in original

    def test_preamble_before_first_heading_uses_default_title(self):
        sections = split_sections("说明文字。\n\n## 规则\n\n内容", default_title="平台说明")

        assert _titles(sections) == ["平台说明", "规则"]
        assert sections[0]["text"] == "说明文字。"
        assert sections[0]["level"] == 0

    def test_document_without_headings_is_one_section(self):
        sections = split_sections("只有一段正文。", default_title="notes")

        assert _titles(sections) == ["notes"]

    def test_headings_inside_code_fences_are_ignored(self):
        text = "# 指南\n\n示例：\n\n```md\n# 不是标题\n## 也不是\n```\n\n## 下一节\n\n正文"
        sections = split_sections(text)

        assert _titles(sections) == ["指南", "指南 > 下一节"]
        assert "# 不是标题" in sections[0]["text"]

    def test_closing_hashes_and_non_headings(self):
        text = "# 标题 ##\n\n#没有空格不是标题\n\n####### 七级不是标题"
        sections = split_sections(text)

        assert _titles(sections) == ["标题"]
        assert "#没有空格不是标题" in sections[0]["text"]

    def test_crlf_and_bom_are_normalized(self, tmp_path):
        path = tmp_path / "rules" / "crlf.md"
        path.parent.mkdir()
        path.write_bytes("﻿# 计费\r\n\r\n## 时长\r\n\r\n1–8 小时\r\n".encode("utf-8"))

        doc = MarkdownLoader().load(path)

        assert "\r" not in doc.text
        assert doc.metadata["title"] == "计费"
        assert _titles(doc.metadata["sections"]) == ["计费 > 时长"]


# ── Document metadata ────────────────────────────────────────────────


class TestMarkdownMetadata:

    def test_required_metadata(self, refund_doc):
        doc = MarkdownLoader().load(refund_doc)

        assert doc.metadata["source_path"] == str(refund_doc.resolve())
        assert doc.metadata["source"] == "refund.md"
        assert doc.metadata["collection"] == "platform_rules"
        assert doc.metadata["doc_type"] == "markdown"
        assert doc.metadata["title"] == "取消与退款"
        assert doc.id.startswith("doc_") and len(doc.id) == 20

    def test_source_is_relative_to_base_dir(self, refund_doc, tmp_path):
        doc = MarkdownLoader(base_dir=tmp_path / "kb").load(refund_doc)

        assert doc.metadata["source"] == "platform_rules/refund.md"

    def test_explicit_collection_overrides_directory(self, refund_doc):
        doc = MarkdownLoader(collection="rules_v2").load(refund_doc)

        assert doc.metadata["collection"] == "rules_v2"

    def test_title_falls_back_to_file_stem(self, tmp_path):
        path = _write(tmp_path / "x" / "ranks.md", "## 段位\n\n黄金")

        assert MarkdownLoader().load(path).metadata["title"] == "ranks"

    def test_same_content_gives_same_id(self, tmp_path):
        a = _write(tmp_path / "a" / "one.md", REFUND_MD)
        b = _write(tmp_path / "b" / "two.md", REFUND_MD)

        assert MarkdownLoader().load(a).id == MarkdownLoader().load(b).id

    def test_rejects_non_markdown_and_missing_files(self, tmp_path):
        txt = _write(tmp_path / "notes.txt", "# 标题")
        with pytest.raises(ValueError, match="not a Markdown"):
            MarkdownLoader().load(txt)
        with pytest.raises(FileNotFoundError):
            MarkdownLoader().load(tmp_path / "missing.md")


# ── Loader factory ───────────────────────────────────────────────────


class TestLoaderFactory:

    @pytest.mark.parametrize("name, expected", [
        ("rules.md", MarkdownLoader),
        ("RULES.MD", MarkdownLoader),
        ("guide.markdown", MarkdownLoader),
        ("report.pdf", PdfLoader),
    ])
    def test_selects_loader_by_extension(self, name, expected):
        assert LoaderFactory.get_provider(name) is expected

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type: '.docx'"):
            LoaderFactory.create("contract.docx")

    def test_supported_extensions_include_md_and_pdf(self):
        assert {".md", ".pdf"} <= set(LoaderFactory.supported_extensions())

    def test_register_rejects_non_loader(self):
        with pytest.raises(ValueError, match="must inherit from BaseLoader"):
            LoaderFactory.register_provider(".txt", object)  # type: ignore[arg-type]

    def test_routing_loader_passes_options_and_reuses_instances(self, refund_doc):
        router = ExtensionRoutingLoader({"md": {"collection": "platform_rules_test"}})

        doc = router.load(refund_doc)

        assert doc.metadata["collection"] == "platform_rules_test"
        assert router.loader_for("other.md") is router.loader_for(refund_doc)
        assert isinstance(router.loader_for("a.pdf"), PdfLoader)
        assert isinstance(router, BaseLoader)


# ── Section-aware chunking ───────────────────────────────────────────


class ParagraphSplitter(BaseSplitter):
    """Splits on blank lines; stands in for the configured splitter."""

    def __init__(self, **kwargs):
        pass

    def split_text(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


def _chunker(splitter: BaseSplitter | None = None) -> DocumentChunker:
    chunker = DocumentChunker.__new__(DocumentChunker)
    chunker._settings = None
    chunker._splitter = splitter or WholeSplitter()
    return chunker


class WholeSplitter(BaseSplitter):
    def __init__(self, **kwargs):
        pass

    def split_text(self, text: str) -> list[str]:
        return [text]


class TestSectionChunking:

    def test_one_chunk_per_section_with_title_path(self, refund_doc):
        doc = MarkdownLoader().load(refund_doc)

        chunks = _chunker().split_document(doc)

        assert [c.metadata["title_path"] for c in chunks] == _titles(doc.metadata["sections"])
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2, 3]
        assert chunks[1].metadata["collection"] == "platform_rules"
        assert chunks[1].metadata["source_ref"] == doc.id
        assert "sections" not in chunks[1].metadata

    def test_chunks_never_cross_sections(self, refund_doc):
        chunks = _chunker().split_document(MarkdownLoader().load(refund_doc))

        assert "全额退款。" in chunks[1].text
        assert "退款 50%" not in chunks[1].text

    def test_continuation_fragments_keep_heading_prefix(self, refund_doc):
        doc = MarkdownLoader().load(refund_doc)

        chunks = _chunker(ParagraphSplitter()).split_document(doc)
        refund_tier = [c for c in chunks if c.metadata["section_index"] == 2]

        # heading block, then body — the body fragment is re-prefixed
        assert refund_tier[-1].text == (
            "# 取消与退款\n## 退款档位\n### 开局前 3–24 小时\n\n退款 50%。"
        )
        assert len({c.id for c in chunks}) == len(chunks)

    def test_documents_without_sections_use_plain_splitting(self):
        from src.core.types import Document

        doc = Document(id="doc_pdf", text="第一段\n\n第二段", metadata={"source_path": "a.pdf"})

        chunks = _chunker(ParagraphSplitter()).split_document(doc)

        assert [c.text for c in chunks] == ["第一段", "第二段"]
        assert "title_path" not in chunks[0].metadata


# ── Incremental ingestion ────────────────────────────────────────────


def _pipeline(tmp_path: Path) -> object:
    """Pipeline with real integrity checker, loader and chunker; mocked storage."""

    class FP:
        collection = "platform_rules"
        force = False

    fp = FP()
    fp.integrity_checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "history.db"))
    fp.loader = ExtensionRoutingLoader({".md": {"collection": "platform_rules"}})
    fp.loader.load = MagicMock(side_effect=fp.loader.load)
    fp.chunker = _chunker()

    passthrough = lambda chunks, trace=None: chunks  # noqa: E731
    for name in ("chunk_refiner", "metadata_enricher", "image_captioner"):
        stage = MagicMock()
        stage.transform.side_effect = passthrough
        setattr(fp, name, stage)

    def process(chunks, trace=None):
        result = MagicMock()
        result.dense_vectors = [[0.1, 0.2]] * len(chunks)
        result.sparse_stats = [{"chunk_id": c.id} for c in chunks]
        return result

    fp.batch_processor = MagicMock()
    fp.batch_processor.process.side_effect = process
    fp.vector_upserter = MagicMock()
    fp.vector_upserter.upsert.side_effect = lambda chunks, vectors, trace=None: [c.id for c in chunks]
    fp.bm25_indexer = MagicMock()
    fp.image_storage = MagicMock()
    return fp


class TestIncrementalIngestion:

    def test_unchanged_markdown_is_skipped_on_reimport(self, refund_doc, tmp_path):
        fp = _pipeline(tmp_path)

        first = IngestionPipeline.run(fp, str(refund_doc))
        second = IngestionPipeline.run(fp, str(refund_doc))

        assert first.success and first.chunk_count == 4
        assert second.success
        assert second.stages["integrity"] == {"skipped": True, "reason": "already_processed"}
        assert fp.loader.load.call_count == 1
        fp.integrity_checker.close()

    def test_changed_markdown_is_reprocessed(self, refund_doc, tmp_path):
        fp = _pipeline(tmp_path)
        IngestionPipeline.run(fp, str(refund_doc))

        refund_doc.write_text(REFUND_MD.replace("50%", "60%"), encoding="utf-8")
        result = IngestionPipeline.run(fp, str(refund_doc))

        assert result.stages["integrity"]["skipped"] is False
        assert fp.loader.load.call_count == 2
        stored: list[Chunk] = fp.vector_upserter.upsert.call_args.args[0]
        assert any("60%" in c.text for c in stored)
        fp.integrity_checker.close()
