import os
search_dir = "C:/Users/10169062/Desktop/AU_IndusMaintdb/modules/search"
print("Files in search directory:")
for file in os.listdir(search_dir):
    if file.endswith('.py'):
        print(f"  - {file}")