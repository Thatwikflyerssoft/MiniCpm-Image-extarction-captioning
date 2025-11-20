#!/usr/bin/env python3
"""
PDF to Markdown Extraction Pipeline

Converts PDFs (local files or URLs) to Markdown with extracted figures using Docling.

Pipeline:
1. Extract markdown and images from PDF using Docling
2. Save markdown and images to output directory
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import services
from services import extract_pdf_with_docling


def process_pdf(
    pdf_source: str,
    output_markdown_path: str,
    temp_dir: str = "./temp_extraction",
    extract_markdown: bool = True
) -> None:
    """
    Extract PDF to Markdown with embedded images using Docling.

    Args:
        pdf_source: Path to input PDF or URL (e.g., "https://arxiv.org/pdf/2408.09869")
        output_markdown_path: Path to save output markdown (ignored if extract_markdown=False)
        temp_dir: Temporary directory for intermediate files
        extract_markdown: If True, save markdown. If False, only extract images.
    """
    try:
        # Create temporary directory
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        if extract_markdown:
            logger.info("🚀 PDF to Markdown Extraction Pipeline")
        else:
            logger.info("🚀 PDF Images Extraction Pipeline")
        logger.info("=" * 70)

        # Extract markdown and images from PDF
        logger.info(f"\n📥 Step 1: Extracting PDF from source...")
        markdown_content, image_dict = extract_pdf_with_docling(
            pdf_source,
            str(temp_path),
            extract_markdown=extract_markdown
        )

        # Save markdown output (if requested)
        if extract_markdown:
            logger.info(f"\n💾 Step 2: Saving markdown output...")
            output_path = Path(output_markdown_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"  ✓ Markdown file saved")

        # Summary
        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ Extraction Complete!")
        logger.info(f"{'=' * 70}")
        logger.info(f"📊 Summary:")
        if extract_markdown:
            logger.info(f"  • Markdown: {output_markdown_path}")
        logger.info(f"  • Images extracted: {len(image_dict)}")
        if image_dict:
            logger.info(f"  • Image directory: {temp_path}/images")
        logger.info(f"{'=' * 70}\n")

    except Exception as e:
        logger.error(f"\n❌ Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_markdown_with_captions.py <pdf_path_or_url> [output_markdown_path] [--images-only]")
        print("\nExamples:")
        print("  Extract markdown + images:")
        print("    python pdf_to_markdown_with_captions.py document.pdf output.md")
        print("    python pdf_to_markdown_with_captions.py https://arxiv.org/pdf/2408.09869 output.md")
        print("\n  Extract images only (for caption generation):")
        print("    python pdf_to_markdown_with_captions.py document.pdf --images-only")
        print("    python pdf_to_markdown_with_captions.py https://arxiv.org/pdf/2408.09869 --images-only")
        sys.exit(1)

    pdf_input = sys.argv[1]

    # Check for --images-only flag
    images_only = "--images-only" in sys.argv

    # Get markdown output path (skip if --images-only is passed)
    if images_only:
        md_output = "output.md"  # Dummy path, won't be used
    else:
        md_output = sys.argv[2] if len(sys.argv) > 2 else "output.md"

    process_pdf(pdf_input, md_output, extract_markdown=not images_only)
