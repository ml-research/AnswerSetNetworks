from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

    # Access the environment variables
    hf_home = os.getenv('HF_HOME')
    huggingface_hub_cache = os.getenv('HUGGINGFACE_HUB_CACHE')
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb_project = os.getenv('WANDB_PROJECT')

    print("-----------------------------")
    print("Environment variables loaded:")
    print("HF_HOME",hf_home)
    print("HUGGINGFACE_HUB_CACHE",huggingface_hub_cache)
    #print("WANDB_API_KEY",wandb_api_key
    print("WANDB_PROJECT",wandb_project)
    print("-----------------------------")