import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def get_user_input():
    """Prompt user for training parameters"""
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING - INTERACTIVE SETUP")
    print("=" * 60)
    print()

    # Get max examples
    print("Your dataset has ~28 million examples.")
    print("Training on CPU with all examples would take days/weeks.")
    print()
    print("Recommended sizes:")
    print("  - Fast test: 10,000-50,000 examples (~30min-2hrs)")
    print("  - Small model: 100,000-200,000 examples (~2-6hrs)")
    print("  - Medium model: 500,000-1,000,000 examples (~6-24hrs)")
    print()

    while True:
        max_examples = input("How many examples do you want to train on? (default: 100000): ").strip()
        if not max_examples:
            max_examples = 100000
            break
        try:
            max_examples = int(max_examples)
            if max_examples < 1000:
                print("Please enter at least 1,000 examples")
                continue
            if max_examples > 5000000:
                confirm = input(f"Warning: {max_examples:,} is very large. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Please enter a valid number")

    print()

    # Get epochs
    while True:
        epochs = input("How many epochs? (default: 2): ").strip()
        if not epochs:
            epochs = 2
            break
        try:
            epochs = int(epochs)
            if epochs < 1 or epochs > 10:
                print("Please enter between 1-10 epochs")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    print()

    # Get batch size
    while True:
        batch_size = input("Batch size? (default: 16 for low memory, 32 for normal): ").strip()
        if not batch_size:
            batch_size = 16
            break
        try:
            batch_size = int(batch_size)
            if batch_size < 4 or batch_size > 128:
                print("Please enter between 4-128")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    print()

    # Use low memory mode?
    low_memory = input("Enable low-memory mode? (recommended if <4GB RAM available) (y/n, default: y): ").strip()
    low_memory = low_memory.lower() != 'n'

    print()
    print("=" * 60)
    print("TRAINING CONFIGURATION:")
    print("=" * 60)
    print(f"  Labels: drawings, parts")
    print(f"  Training examples: {max_examples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Low memory mode: {'Yes' if low_memory else 'No'}")
    print(f"  Workers: {'0 (single-threaded)' if low_memory else 'Auto (2-8 based on system)'}")
    print(f"  Device: CPU")
    print("=" * 60)
    print()

    confirm = input("Start training with these settings? (y/n): ").strip()
    if confirm.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)

    return {
        'max_examples': max_examples,
        'epochs': epochs,
        'batch_size': batch_size,
        'low_memory': low_memory
    }


if __name__ == "__main__":
    # Get user preferences
    config = get_user_input()

    # Build command line arguments
    sys.argv = [
        'run_training.py',
        '--labels', 'drawings', 'parts',
        '--max-examples', str(config['max_examples']),
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--no-cache' if config['low_memory'] else '--use-cache',
    ]

    # Only set workers to 0 if low memory mode
    if config['low_memory']:
        sys.argv.extend(['--num-workers', '0'])
    # Otherwise let it auto-detect (2-8 workers based on CPU/RAM)

    print()
    print("Starting training...")
    print()

    # Import and run the training script
    from modules.emtac_ai.training_scripts.dataset_intent_train.cld_adpt_strm_train_intent import main

    main()