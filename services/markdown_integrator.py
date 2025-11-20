"""Markdown integration service for combining captions with content."""
# File needs cleanup - added temporarily

from typing import Dict
import logging

logger = logging.getLogger(__name__)


def integrate_captions_into_markdown(
    markdown_content: str,
    image_dict: Dict[str, str],
    captions: Dict[str, str],
    images_relative_path: str = "images"
) -> str:
    """
    Integrate image captions into markdown content.

    Args:
        markdown_content: Original markdown content from Docling
        image_dict: Mapping of image names to paths
        captions: Mapping of image names to their captions
        images_relative_path: Relative path for image references in markdown

    Returns:
        Enhanced markdown with captions
    """
    enhanced_markdown = markdown_content

    for image_name, image_path in image_dict.items():
        # Create markdown image reference
        relative_image_path = f"{images_relative_path}/{image_name}"
        caption = captions.get(image_name, "[No caption available]")

        # Format: ![image](path)\nCaption: description
        image_markdown = f"![{image_name}]({relative_image_path})\n\n**Figure:** {caption}\n"

        logger.info(f"Integrated caption for: {image_name}")

    # Add metadata section
    metadata = f"""# Document Metadata

- Total images processed: {len(image_dict)}
- Images with captions: {len([c for c in captions.values() if c != '[Caption generation failed]'])}

---

"""

    enhanced_markdown = metadata + enhanced_markdown

    return enhanced_markdown
