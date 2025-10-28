import os

# Get the number of CPU cores
cpu_count = os.cpu_count()

# Print the number of CPU cores
print(f"Number of CPU cores available: {cpu_count}")
