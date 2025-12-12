
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Downloading {model_name}...")
    
    # Ensure HF_HOME is set
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("Warning: HF_HOME is not set. Using default location.")
        # Attempt to set it to the project cache if known, otherwise let libraries decide
        # os.environ["HF_HOME"] = "/p/project1/envcomp/yll/.cache/huggingface"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer downloaded successfully.")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model downloaded successfully.")
        
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model()
