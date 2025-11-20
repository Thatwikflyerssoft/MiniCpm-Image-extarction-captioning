#!/usr/bin/env python3
"""
PDF to Markdown Extraction Pipeline with Automatic Image Captioning

Converts PDFs (local files or URLs) to Markdown with extracted figures using Docling,
then automatically generates captions for all extracted images using MiniCPM-V.

Pipeline:
1. Extract markdown and images from PDF using Docling
2. Save markdown and images to output directory
3. Generate captions for all extracted images using MiniCPM-V
"""

import os
import sys
from pathlib import Path
import logging
from PIL import Image
import torch
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import services
from services import extract_pdf_with_docling

# Set your Hugging Face token
HF_TOKEN = "your hf token"
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_caption_model():
    """Load MiniCPM-V 2.6 model with CPU offloading for 6GB GPU"""
    from transformers import AutoModel, AutoTokenizer
    
    logger.info("Loading MiniCPM-V 2.6 model for caption generation...")
    logger.info("Note: Using CPU+GPU hybrid mode for 6GB GPU\n")
    
    model_name = "openbmb/MiniCPM-V-2_6"
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Load model with automatic device mapping (will use CPU+GPU)
    logger.info("Loading model (this will take a few minutes)...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto",  # Automatically splits between CPU and GPU
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "5GB", "cpu": "16GB"}  # Limit GPU to 5GB, rest to CPU
    )
    
    model.eval()
    
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info("✓ Model loaded successfully!")
        logger.info(f"✓ GPU Memory used: {allocated:.2f} GB")
        logger.info("✓ Some layers on CPU, some on GPU (hybrid mode)\n")
    
    return model, tokenizer, device

def generate_caption(model, tokenizer, image_path, prompt="Describe this image in detail."):
    """Generate caption for a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Generate caption with memory optimization
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        
        with torch.inference_mode():  # More memory efficient than torch.no_grad()
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,  # Faster, deterministic
                max_new_tokens=512
            )
        
        # Clear cache after each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return res
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def generate_image_captions(image_folder, output_file="captions.txt", prompt="Describe this image in detail."):
    """Generate captions for all images in the extracted images folder"""
    
    logger.info("🖼️ Starting image caption generation...")
    
    # Load model once
    model, tokenizer, device = load_caption_model()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_folder_path = Path(image_folder)
    
    if not image_folder_path.exists():
        logger.error(f"Image folder '{image_folder}' does not exist!")
        return []
    
    image_files = sorted([f for f in image_folder_path.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    if not image_files:
        logger.info(f"No images found in '{image_folder}'")
        return []
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        caption = generate_caption(model, tokenizer, image_path, prompt)
        
        if caption:
            results.append({
                'filename': image_path.name,
                'caption': caption
            })
            logger.info(f"✓ Caption generated: {caption[:120]}...")
            
            # Show memory usage every 5 images
            if device == "cuda" and idx % 5 == 0:
                mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"  [GPU Memory: {mem_used:.2f} GB]")
        else:
            logger.warning(f"✗ Failed to generate caption for {image_path.name}")
    
    # Save results
    logger.info(f"Saving captions to '{output_file}'...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Image Captions Report\n")
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Prompt Used: {prompt}\n")
        f.write("=" * 70 + "\n\n")
        
        for idx, result in enumerate(results, 1):
            f.write(f"{idx}. FILE: {result['filename']}\n")
            f.write(f"   CAPTION: {result['caption']}\n")
            f.write("-" * 70 + "\n\n")
    
    logger.info(f"✓ Successfully generated captions for {len(results)}/{len(image_files)} images")
    logger.info(f"✓ Captions saved to: {output_file}")
    
    return results

def process_pdf_with_captions(
    pdf_source: str,
    output_markdown_path: str,
    temp_dir: str = "./temp_extraction",
    extract_markdown: bool = True,
    generate_captions: bool = True,
    caption_prompt: str = "Describe this image in detail."
) -> None:
    """
    Extract PDF to Markdown with embedded images using Docling, then generate captions.

    Args:
        pdf_source: Path to input PDF or URL (e.g., "https://arxiv.org/pdf/2408.09869")
        output_markdown_path: Path to save output markdown (ignored if extract_markdown=False)
        temp_dir: Temporary directory for intermediate files
        extract_markdown: If True, save markdown. If False, only extract images.
        generate_captions: If True, generate captions for extracted images
        caption_prompt: Prompt to use for caption generation
    """
    try:
        # Create temporary directory
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        if extract_markdown:
            logger.info("🚀 PDF to Markdown Extraction Pipeline with Image Captioning")
        else:
            logger.info("🚀 PDF Images Extraction Pipeline with Image Captioning")
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
            logger.info(f"  ✓ Markdown file saved: {output_markdown_path}")

        # Generate captions for extracted images (if requested and images exist)
        image_folder = temp_path / "images"
        caption_results = []
        
        if generate_captions and image_dict:
            logger.info(f"\n🖼️ Step 3: Generating captions for {len(image_dict)} images...")
            caption_output = temp_path / "image_captions.txt"
            caption_results = generate_image_captions(
                image_folder=str(image_folder),
                output_file=str(caption_output),
                prompt=caption_prompt
            )
        elif generate_captions and not image_dict:
            logger.info("\nℹ️ No images found to generate captions for.")
        elif not generate_captions:
            logger.info("\n⏭️ Skipping caption generation as requested.")

        # Summary
        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ Pipeline Complete!")
        logger.info(f"{'=' * 70}")
        logger.info(f"📊 Summary:")
        if extract_markdown:
            logger.info(f"  • Markdown: {output_markdown_path}")
        logger.info(f"  • Images extracted: {len(image_dict)}")
        if image_dict:
            logger.info(f"  • Image directory: {image_folder}")
        if generate_captions and caption_results:
            logger.info(f"  • Captions generated: {len(caption_results)}")
            logger.info(f"  • Captions file: {temp_path / 'image_captions.txt'}")
        logger.info(f"{'=' * 70}\n")

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_markdown_with_captions.py <pdf_path_or_url> [output_markdown_path] [--images-only] [--no-captions]")
        print("\nExamples:")
        print("  Extract markdown + images + captions (default):")
        print("    python pdf_to_markdown_with_captions.py document.pdf output.md")
        print("    python pdf_to_markdown_with_captions.py https://arxiv.org/pdf/2408.09869 output.md")
        print("\n  Extract images only with captions:")
        print("    python pdf_to_markdown_with_captions.py document.pdf --images-only")
        print("\n  Extract markdown + images without captions:")
        print("    python pdf_to_markdown_with_captions.py document.pdf output.md --no-captions")
        print("\n  Extract images only without captions:")
        print("    python pdf_to_markdown_with_captions.py document.pdf --images-only --no-captions")
        sys.exit(1)

    pdf_input = sys.argv[1]

    # Check for flags
    images_only = "--images-only" in sys.argv
    no_captions = "--no-captions" in sys.argv

    # Get markdown output path (skip if --images-only is passed)
    if images_only:
        md_output = "output.md"  # Dummy path, won't be used
    else:
        # Find the first non-flag argument after pdf_input as output path
        md_output = "output.md"  # default
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                md_output = arg
                break

    process_pdf_with_captions(
        pdf_input, 
        md_output, 
        extract_markdown=not images_only,
        generate_captions=not no_captions
    )