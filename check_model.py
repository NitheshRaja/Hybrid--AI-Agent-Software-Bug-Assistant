import os
from pathlib import Path

model_path = Path("D:/software-bug-assistant/models/gemma-2-2b-it-Q4_K_M.gguf")

result = []
result.append(f"Model path: {model_path}")
result.append(f"Exists: {model_path.exists()}")

if model_path.exists():
    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    result.append(f"Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    
    if size_gb >= 1.4:
        result.append("Status: DOWNLOAD COMPLETE!")
    else:
        result.append(f"Status: Still downloading... (expected ~1.5 GB)")
else:
    result.append("Status: File not found")

# Write results to file
with open("D:/software-bug-assistant/model_status.txt", "w") as f:
    f.write("\n".join(result))
    
print("\n".join(result))







