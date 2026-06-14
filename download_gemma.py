"""Download Gemma-2B GGUF model for local inference"""

import urllib.request
from pathlib import Path

MODEL_DIR = Path("models")
GEMMA_2B_FILENAME = "gemma-2-2b-it-Q4_K_M.gguf"
GEMMA_2B_URL = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"


def download_gemma_model():
    """Download Gemma-2B GGUF model from HuggingFace"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / GEMMA_2B_FILENAME
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"File size: {file_size:.2f} GB")
        return str(model_path)
    
    print(f"Downloading Gemma-2B ({GEMMA_2B_FILENAME})...")
    print(f"URL: {GEMMA_2B_URL}")
    print(f"Destination: {model_path}")
    print("This may take a few minutes (~1.5GB)...")
    print("")
    
    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        percent = int(downloaded * 100 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\rProgress: {percent}% ({downloaded_mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(GEMMA_2B_URL, model_path, progress_hook)
        print(f"\n\nDownload complete!")
        print(f"Model saved to: {model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"File size: {file_size:.2f} GB")
        return str(model_path)
    except Exception as e:
        print(f"\n\nDownload failed: {e}")
        print("\nYou can manually download from:")
        print(f"  {GEMMA_2B_URL}")
        print(f"And save to: {model_path}")
        return None


if __name__ == "__main__":
    download_gemma_model()







