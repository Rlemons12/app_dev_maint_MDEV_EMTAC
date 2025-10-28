import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the training script
from modules.emtac_ai.training_scripts.dataset_intent_train.cld_adpt_strm_train_intent import main

if __name__ == "__main__":
    main()