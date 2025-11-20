"""PDF extraction service using Docling."""

import sys
from pathlib import Path
from typing import Tuple, Dict
import logging

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


def _setup_image_directory(output_dir: str) -> Path:
    """
    Create and return the image output directory.

    Args:
        output_dir: Base output directory

    Returns:
        Path to the images subdirectory
    """
    image_dir = Path(output_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    logger.info("✓ Output directory created")
    return image_dir


def _create_converter():
    """
    Create a DocumentConverter with optimized pipeline options for image extraction.

    Returns:
        DocumentConverter instance configured for image extraction

    Raises:
        SystemExit: If Docling import fails
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as e:
        logger.error(f"Failed to import Docling: {e}")
        logger.error("Docling not installed. Install with: pip install docling")
        sys.exit(1)

    logger.info("⏳ [1/3] Initializing Docling converter...")

    # Configure pipeline options for image extraction
    pdf_options = PdfPipelineOptions()
    pdf_options.generate_picture_images = True  # Extract embedded pictures
    pdf_options.generate_page_images = True     # Generate page images as fallback
    pdf_options.do_picture_classification = True  # Classify images
    pdf_options.images_scale = 2.0  # Better quality resolution

    # Create format option with pipeline options
    pdf_format_option = PdfFormatOption(pipeline_options=pdf_options)
    format_options = {InputFormat.PDF: pdf_format_option}

    return DocumentConverter(format_options=format_options)


def _convert_pdf_to_markdown(pdf_source: str, converter, extract_markdown: bool) -> str:
    """
    Convert PDF to markdown using Docling.

    Args:
        pdf_source: Path to input PDF file or URL
        converter: DocumentConverter instance
        extract_markdown: If True, export to markdown format

    Returns:
        Markdown content string (empty if extract_markdown=False)
    """
    logger.info("⏳ [2/3] Downloading and processing PDF (this may take a moment)...")
    result = converter.convert(pdf_source)

    markdown_content = ""
    if extract_markdown:
        logger.info("⏳ [3/3] Exporting to markdown format...")
        markdown_content = result.document.export_to_markdown()
        logger.info("✓ Markdown conversion complete")
    else:
        logger.info("⏳ [3/3] Skipping markdown export (images only mode)")

    return markdown_content, result


def _save_image_to_disk(image, idx: int, image_dir: Path) -> Tuple[str, str]:
    """
    Save a single image to disk.

    Args:
        image: PIL Image object to save
        idx: Image index number
        image_dir: Directory to save image

    Returns:
        Tuple of (image_name, image_path)
    """
    image_name = f"figure_{idx:03d}.png"
    image_path = image_dir / image_name
    image.save(str(image_path))
    logger.info(f"  ✓ Saved image: {image_name}")
    return image_name, str(image_path)


def _extract_single_picture(picture, result, idx: int, image_dir: Path) -> Tuple[str, str]:
    """
    Extract a single picture from the document and save it.

    Args:
        picture: Picture object from document
        result: ConversionResult from Docling
        idx: Image index number
        image_dir: Directory to save image

    Returns:
        Tuple of (image_name, image_path) or (None, None) if extraction failed
    """
    try:
        image = picture.get_image(result.document)

        if image:
            return _save_image_to_disk(image, idx, image_dir)
        else:
            logger.warning(f"  ✗ Failed to extract image {idx} (returned None)")
            return None, None
    except Exception as e:
        logger.warning(f"  ✗ Error extracting image {idx}: {e}")
        return None, None


def _extract_all_pictures(result, image_dir: Path) -> Dict[str, str]:
    """
    Extract all pictures from the document.

    Args:
        result: ConversionResult from Docling
        image_dir: Directory to save images

    Returns:
        Dictionary mapping image names to their file paths
    """
    logger.info(f"\n📸 Extracting {len(result.document.pictures)} image(s)...")
    image_dict = {}

    for idx, picture in enumerate(result.document.pictures, 1):
        image_name, image_path = _extract_single_picture(picture, result, idx, image_dir)
        if image_name and image_path:
            image_dict[image_name] = image_path

    return image_dict


def _log_extraction_summary(image_dict: Dict[str, str]) -> None:
    """
    Log the summary of image extraction results.

    Args:
        image_dict: Dictionary of extracted images
    """
    if image_dict:
        logger.info(f"✓ Successfully extracted {len(image_dict)} image(s)")
    else:
        logger.warning("⚠ No images could be extracted from document")


def _extract_images(result, image_dir: Path) -> Dict[str, str]:
    """
    Extract images from PDF document and save to disk.

    Args:
        result: ConversionResult from Docling
        image_dir: Path to save images

    Returns:
        Dictionary mapping image names to their file paths
    """
    if not (hasattr(result.document, 'pictures') and result.document.pictures):
        logger.info("\n📸 No images found in document")
        return {}

    image_dict = _extract_all_pictures(result, image_dir)
    _log_extraction_summary(image_dict)

    return image_dict


def _link_images_in_markdown(
    markdown_content: str,
    image_dict: Dict[str, str]
) -> str:
    """
    Replace image placeholders in markdown with actual image references.

    Args:
        markdown_content: Original markdown content
        image_dict: Mapping of image names to file paths

    Returns:
        Updated markdown content with image references
    """
    if not image_dict:
        return markdown_content

    logger.info("\n🔗 Linking images in markdown...")
    markdown_lines = markdown_content.split('\n')
    image_idx = 0
    image_names = list(image_dict.keys())

    updated_lines = []
    for line in markdown_lines:
        if '<!-- image -->' in line and image_idx < len(image_names):
            image_name = image_names[image_idx]
            relative_path = f"images/{image_name}"
            markdown_image = f"![{image_name}]({relative_path})"
            updated_lines.append(markdown_image)
            image_idx += 1
        else:
            updated_lines.append(line)

    markdown_content = '\n'.join(updated_lines)
    logger.info(f"  ✓ Linked {image_idx} image(s) to markdown")

    return markdown_content


def extract_pdf_with_docling(
    pdf_source: str,
    output_dir: str,
    extract_markdown: bool = True
) -> Tuple[str, Dict[str, str]]:
    """
    Extract markdown and figures from PDF using Docling.

    Args:
        pdf_source: Path to input PDF file or URL (e.g., "https://arxiv.org/pdf/2408.09869")
        output_dir: Directory to save extracted images
        extract_markdown: If True, export content as markdown. If False, skip markdown export.

    Returns:
        Tuple of (markdown_content, image_dict)
        markdown_content: Empty string if extract_markdown=False
        image_dict: {'image_name': 'path_to_image'}
    """
    logger.info(f"📄 Source: {pdf_source}")

    # Setup output directory
    image_dir = _setup_image_directory(output_dir)

    # Create converter with optimized options
    converter = _create_converter()

    # Convert PDF to markdown
    markdown_content, result = _convert_pdf_to_markdown(pdf_source, converter, extract_markdown)

    # Extract images from document
    image_dict = _extract_images(result, image_dir)

    # Link extracted images in markdown
    if extract_markdown:
        markdown_content = _link_images_in_markdown(markdown_content, image_dict)

    return markdown_content, image_dict
