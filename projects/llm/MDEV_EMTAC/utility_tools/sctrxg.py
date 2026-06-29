import os
from dotenv import load_dotenv

load_dotenv()

print("MODEL_PATH_CLIP =", os.getenv("MODEL_PATH_CLIP"))
print("MODELS_CLIP_DIR =", os.getenv("MODELS_CLIP_DIR"))
