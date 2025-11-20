"""Services package for PDF processing pipeline."""

from .pdf_extractor import extract_pdf_with_docling

__all__ = [
    "extract_pdf_with_docling",
]
