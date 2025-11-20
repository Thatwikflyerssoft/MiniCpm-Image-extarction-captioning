import os
from pathlib import Path
from PIL import Image
import torch
import gc

# Set your Hugging Face token
HF_TOKEN = "your hf token"
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_model():
    """Load MiniCPM-V 2.6 model with CPU offloading for 6GB GPU"""
    from transformers import AutoModel, AutoTokenizer
    
    print("Loading MiniCPM-V 2.6 model...")
    print("Note: Using CPU+GPU hybrid mode for 6GB GPU\n")
    
    model_name = "openbmb/MiniCPM-V-2_6"
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Load model with automatic device mapping (will use CPU+GPU)
    print("\nLoading model (this will take a few minutes)...")
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
        print(f"\n✓ Model loaded successfully!")
        print(f"✓ GPU Memory used: {allocated:.2f} GB")
        print(f"✓ Some layers on CPU, some on GPU (hybrid mode)\n")
    
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
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_images(image_folder, output_file="captions.txt", prompt="Describe this image in detail."):
    """Process all images in a folder and save captions"""
    
    # Load model once
    model, tokenizer, device = load_model()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_folder = Path(image_folder)
    
    if not image_folder.exists():
        print(f"Error: Folder '{image_folder}' does not exist!")
        return []
    
    image_files = sorted([f for f in image_folder.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print(f"No images found in '{image_folder}'")
        return []
    
    print(f"Found {len(image_files)} images to process")
    print("=" * 70)
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        caption = generate_caption(model, tokenizer, image_path, prompt)
        
        if caption:
            results.append({
                'filename': image_path.name,
                'caption': caption
            })
            print(f"✓ Caption: {caption[:120]}...")
            
            # Show memory usage every 5 images
            if device == "cuda" and idx % 5 == 0:
                mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"  [GPU Memory: {mem_used:.2f} GB]")
        else:
            print(f"✗ Failed to generate caption")
    
    # Save results
    print("\n" + "=" * 70)
    print(f"\nSaving results to '{output_file}'...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Image Captions Report\n")
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Prompt Used: {prompt}\n")
        f.write("=" * 70 + "\n\n")
        
        for idx, result in enumerate(results, 1):
            f.write(f"{idx}. FILE: {result['filename']}\n")
            f.write(f"   CAPTION: {result['caption']}\n")
            f.write("-" * 70 + "\n\n")
    
    print(f"✓ Successfully processed {len(results)}/{len(image_files)} images")
    print(f"✓ Results saved to: {output_file}")
    
    return results

def process_single_image(image_path, prompt="Describe this image in detail."):
    """Process a single image and return the caption"""
    print("Processing single image...")
    model, tokenizer, device = load_model()
    caption = generate_caption(model, tokenizer, image_path, prompt)
    return caption

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("MiniCPM-V 2.6 Image Captioning Tool")
    print("=" * 70 + "\n")
    
    # CHANGE THIS to your image folder path
    # Example: r"C:\Users\YourName\Pictures\my_images"
    image_folder = r"C:\Users\thatw\Downloads\pdf_image_extract_captioner\pdf_image_extract_captioner\temp_extraction\images"
    
    # Check if folder exists
    if not Path(image_folder).exists():
        print(f"ERROR: Folder not found: {image_folder}")
        print("\nPlease update line 153 with your correct image folder path.")
        print("Right-click your folder → Copy as path → Paste it in the script")
        input("\nPress Enter to exit...")
        exit()
    
    # Process all images
    try:
        results = process_images(
            image_folder, 
            output_file="captions.txt",
            prompt="Describe this image in detail."
        )
        
        print("\n" + "=" * 70)
        print("DONE! Check 'captions.txt' for results.")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print("\nIf you get memory errors, try:")
        print("1. Close other applications")
        print("2. Process fewer images")
        print("3. Restart your computer")
    
    input("\nPress Enter to exit...")