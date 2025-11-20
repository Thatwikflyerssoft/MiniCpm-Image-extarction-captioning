#!/usr/bin/env python3
"""
Example usage and testing script for the PDF to Markdown pipeline.
"""

import os
from pathlib import Path
from pdf_to_markdown_with_captions import process_pdf


def example_single_pdf():
    """Process a single PDF file."""
    pdf_path = "sample_document.pdf"
    output_path = "sample_document_annotated.md"

    print("Example 1: Processing single PDF")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_path}")
    print()

    if os.path.exists(pdf_path):
        process_pdf(pdf_path, output_path)
    else:
        print(f"Error: {pdf_path} not found. Please provide a sample PDF.")


def example_batch_processing():
    """Process multiple PDFs from a directory."""
    pdf_directory = "./pdfs"
    output_directory = "./markdown_outputs"

    print("\nExample 2: Batch processing PDFs")
    print(f"Input directory: {pdf_directory}")
    print(f"Output directory: {output_directory}")
    print()

    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Process each PDF
    pdf_files = Path(pdf_directory).glob("*.pdf")
    for pdf_file in pdf_files:
        output_md = Path(output_directory) / f"{pdf_file.stem}.md"
        print(f"\nProcessing: {pdf_file.name}")
        process_pdf(str(pdf_file), str(output_md))

    print(f"\n✓ All PDFs processed. Check {output_directory} for outputs.")


def example_with_custom_temp_dir():
    """Process PDF with custom temporary directory."""
    pdf_path = "document.pdf"
    output_path = "document.md"
    custom_temp_dir = "./my_temp_extraction"

    print("\nExample 3: Using custom temporary directory")
    print(f"Temp directory: {custom_temp_dir}")
    print()

    if os.path.exists(pdf_path):
        process_pdf(pdf_path, output_path, temp_dir=custom_temp_dir)
    else:
        print(f"Error: {pdf_path} not found.")


def example_verify_installation():
    """Verify all dependencies are installed."""
    print("\nExample 4: Verifying installation")
    print()

    dependencies = {
        "docling": "PDF extraction",
        "PIL": "Image processing",
        "torch": "Deep learning framework",
        "transformers": "Model management",
    }

    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package:15} - {description}")
        except ImportError:
            print(f"✗ {package:15} - {description} [NOT INSTALLED]")
            all_installed = False

    if all_installed:
        print("\n✓ All dependencies installed successfully!")
    else:
        print("\n✗ Some dependencies are missing. Run: pip install -r requirements.txt")

    return all_installed


if __name__ == "__main__":
    print("=" * 70)
    print("PDF to Markdown with Vision Captions - Examples")
    print("=" * 70)

    # First, verify installation
    if not example_verify_installation():
        print("\n⚠ Please install missing dependencies before running examples.")
        exit(1)

    print("\n" + "=" * 70)
    print("Available Examples:")
    print("=" * 70)
    print("1. Single PDF processing")
    print("2. Batch processing from directory")
    print("3. Custom temporary directory")
    print("\nTo run a specific example, edit this file or use the main script:")
    print("  python pdf_to_markdown_with_captions.py <pdf_path> [output_md_path]")
    print("=" * 70)

    # Uncomment the example you want to run:
    # example_single_pdf()
    # example_batch_processing()
    # example_with_custom_temp_dir()
