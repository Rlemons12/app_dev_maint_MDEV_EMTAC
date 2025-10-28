import os
import csv

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def write_to_spreadsheet(file_list, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file in file_list:
            writer.writerow([file])

def main():
    script_directory = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    default_directory = os.getcwd()  # Default to the current directory
    directory = input(f"Enter the directory path (default is '{default_directory}'): ") or default_directory
    if not os.path.isdir(directory):
        print("The specified directory does not exist.")
        return

    default_output_file = "file_list.csv"  # Default output file name
    output_file = input(f"Enter the output spreadsheet file path (default is '{os.path.join(script_directory, default_output_file)}'): ") or os.path.join(script_directory, default_output_file)

    # Get the list of all files in the directory and its subfolders
    file_list = list_files(directory)

    # Write the file list to a spreadsheet
    write_to_spreadsheet(file_list, output_file)

    print("File names have been written to", output_file)

    choose_again = input("Do you want to choose another output file location? (yes/no): ").lower()
    if choose_again == "yes":
        main()

if __name__ == "__main__":
    main()
