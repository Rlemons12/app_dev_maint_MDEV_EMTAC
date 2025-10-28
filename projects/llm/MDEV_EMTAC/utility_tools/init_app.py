import os
import sys

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations
from modules.configuration.config import directories_to_check

# Call the function to create directories
create_directories(directories_to_check)

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, globals())

def main():
    try:
        print("Running load_active_drawing_list.py...")
        run_script('load_active_drawing_list.py')
        
        print("Running load_keywords_file.py...")
        run_script('load_keywords_file.py')
        
        print("Running load_equipment_relationships_table_data.py...")
        run_script('load_equipment_relationships_table_data.py')
        
        print("Running load_mp2_items_boms.py...")
        run_script('load_mp2_items_boms.py')
        
        print("All data loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()