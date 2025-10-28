import os
import re
import pandas as pd
import sys

# Get the current script directory path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Navigate one level up to locate the requirements.txt file
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Add PARENT_DIR to the Python path
sys.path.append(PARENT_DIR)

def find_definitions_and_usages(root_folder, output_txt_file, output_excel_file):
    def_dict = {}
    pattern_def = re.compile(r'def (\w+)\(')  # To match function definitions
    pattern_call = re.compile(r'(\w+)\(')  # To match function calls

    # Loop through all Python files in the project directory
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(foldername, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for line in lines:
                    # Search for function definitions
                    match_def = pattern_def.search(line)
                    if match_def:
                        func_name = match_def.group(1)
                        if func_name not in def_dict:
                            def_dict[func_name] = []
                        print("Found function definition:", func_name)
                    
                    # Search for function calls
                    match_call = pattern_call.findall(line)
                    if match_call:
                        for call in match_call:
                            if call in def_dict:
                                def_dict[call].append(filepath)
                                print("Found function call:", call, "in file:", filepath)

    # Print debug information
    print("Dictionary of function definitions and usages:")
    print(def_dict)

    # Write to output text file
    with open(output_txt_file, 'w') as f:
        for func, usages in def_dict.items():
            f.write(f"Function: {func}\n")
            f.write("Usages:\n")
            for usage in usages:
                f.write(f"  {usage}\n")
            f.write("\n")

    # Write to output Excel file
    df_data = []
    for func, usages in def_dict.items():
        for usage in usages:
            df_data.append([func, usage])
    
    df = pd.DataFrame(df_data, columns=['Function', 'Usage'])
    df.to_excel(output_excel_file, index=False)

# Get the parent directory of the script file
ROOT_FOLDER = PARENT_DIR

if __name__ == "__main__":
    output_txt_file = 'function_usages.txt'  # The output text file
    output_excel_file = 'function_usages.xlsx'  # The output Excel file
    find_definitions_and_usages(ROOT_FOLDER, output_txt_file, output_excel_file)

