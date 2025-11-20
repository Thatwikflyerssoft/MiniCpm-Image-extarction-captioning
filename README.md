PDF → Markdown + Image Extraction + Vision Captioning Pipeline

A complete end-to-end Python pipeline that:
Converts PDFs to Markdown
Extracts all images from the PDF
Generates AI captions using MiniCPM-V-2.6 (GPU recommended)
Produces a final Markdown file containing:
Extracted text
Linked images
Captions for each image

This repository contains three main Python files that work together to accomplish this.

🔐 Hugging Face Token (Required)

MiniCPM-V-2.6 requires a Hugging Face access token.

Add this line inside your code before the model loads:
from huggingface_hub import login
login("YOUR_HF_TOKEN_HERE")

Add it in either of these files:

test_minicpm_caption.py

or

app.py

This must be placed before the model is loaded so the captioning model can authenticate and download properly.

🚀 Features

PDF → Markdown conversion (Docling)
High-quality image extraction from PDF
Vision caption generation for every image
Full pipeline orchestrator to merge everything
Works with local PDFs and URLs
Clean output folder structure
GPU support for fast inference

📁 Components (Main Files)
1. test_minicpm_caption.py — Vision Caption Model

This script loads the MiniCPM-V-2.6 model and generates a caption for any input image.

Responsibilities:

Load model + tokenizer
Accept image path
Generate caption text
Return caption
Uses GPU if available (recommended)

2. pdf_to_markdown_with_captions.py — PDF Extraction

Handles everything related to Docling:
Responsibilities:
Convert PDF → Markdown
Extract every embedded image
Save images to temp_extraction/images/
Output extracted markdown

No captioning happens here — only PDF parsing + extraction.

3. app.py — Main Orchestrator (Runs Everything)

This file combines both extraction + captioning.
Workflow:

Runs PDF extraction
Collects extracted images
Sends each image to the MiniCPM caption generator
Appends captions below each image in the Markdown
Produces the final_markdown.md

This is the recommended file to run.

⚡ GPU Recommendation

MiniCPM-V-2.6 is a large vision-language model, so:

Device	Speed
GPU	    Fast (recommended)
CPU	    Very slow, not recommended

To ensure GPU usage:

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

🛠 Setup
1. Create Virtual Environment
python3 -m venv ds
source ds/bin/activate

2. Install Dependencies
pip install -r requirements.txt

3. Verify Docling Installation
python3 -c "from docling.document_converter import DocumentConverter; print('✓ Docling is ready!')"

▶️ Usage
✅ Run Full Pipeline (Recommended)

This runs extraction + captioning + final markdown generation:

python3 app.py input.pdf output.md

🧩 Run Individual Files (If Needed)
Extract Only Text + Images
python3 pdf_to_markdown_with_captions.py input.pdf output.md

Caption a Single Image
python3 test_minicpm_caption.py path/to/image.png

📂 Project Structure
pdf_image_extract_captioner/
├── app.py                             # Full orchestrator
├── test_minicpm_caption.py            # MiniCPM model captioning
├── pdf_to_markdown_with_captions.py   # Docling extraction logic
├── services/
│   ├── __init__.py
│   └── pdf_extractor.py               # Core PDF extraction helper
├── requirements.txt
├── README.md
└── temp_extraction/
    └── images/                        # Extracted images

🔄 How It Works (Flow)
1. PDF → Markdown + Images

Using Docling’s pipeline, we extract:

Text as Markdown

Embedded images

2. Each image is passed to MiniCPM

The caption script loads the model and generates descriptions.

3. Captions are inserted into Markdown

app.py merges everything and restructures the output.

4. Final Markdown saved

Includes:

Clean text

Images

AI-generated captions

🪛 Troubleshooting
Docling issues
Ensure Python 3.8+
Install requirements again
Try re-running the import test
MiniCPM model slow
You are likely using CPU → enable GPU
Install CUDA PyTorch
Reduce image resolution if needed
No images extracted
PDF may be pure text
Enable page image fallback in Docling