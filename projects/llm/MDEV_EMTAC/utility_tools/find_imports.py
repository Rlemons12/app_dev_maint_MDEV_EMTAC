import os
import ast
from modules.configuration.config import BASE_DIR

def find_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    return imports

def find_modules(base_dir):
    modules_used = set()
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                imports = find_imports(file_path)
                modules_used.update(imports)
    
    return modules_used

def main():
    modules_used = find_modules(BASE_DIR)
    output_file = os.path.join(BASE_DIR, 'modules.txt')

    with open(output_file, 'w') as file:
        for module in sorted(modules_used):
            file.write(module + '\n')

    print(f"Modules used in your project have been saved to {output_file}")

if __name__ == "__main__":
    main()
