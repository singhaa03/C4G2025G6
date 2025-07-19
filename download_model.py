import os
import requests
import zipfile
from pathlib import Path
import sys

def download_model():
    model_dir = Path("model")
    
    # Check if model already exists
    if model_dir.exists() and any(model_dir.iterdir()):
        print("✅ Model already exists!")
        return
    
    # Replace with your actual Google Drive file ID
    file_id = "1yXXEa8ID-K39qWZnORbPMlputRIW_xL9"
    download_url = f"https://drive.google.com/file/d/1yXXEa8ID-K39qWZnORbPMlputRIW_xL9/view?usp=sharing"
    
    print("📥 Downloading model files...")
    
    try:
        # Download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save to model.zip
        with open("model.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Download complete!")
        
        # Extract the zip file
        print("📁 Extracting model files...")
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up zip file
        os.remove("model.zip")
        
        print("🚀 Model setup complete! You can now run: uvicorn main:app --reload")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading model: {e}")
        print("📎 Manual download link: https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        print("📁 Download model.zip and extract to project root")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("❌ Error: Downloaded file is not a valid zip file")
        print("📎 Please download manually: https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
