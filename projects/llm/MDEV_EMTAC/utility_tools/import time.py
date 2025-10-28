import time
from tkinter import Tk, messagebox
import os
from pywinauto import Desktop

def show_reminder():
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()

    # Display the pop-up message
    messagebox.showinfo("Reminder", "Please verify all information before using this document.")
    
    # Close the tkinter window
    root.destroy()

def monitor_documents():
    # File extensions to monitor
    extensions = ['.doc', '.docx', '.pdf', '.odt', '.txt']

    while True:
        # Get a list of all currently open windows
        windows = Desktop(backend="uia").windows()

        for win in windows:
            try:
                # Get the window title and check if it ends with a document extension
                title = win.window_text().lower()
                if any(title.endswith(ext) for ext in extensions):
                    show_reminder()
                    time.sleep(10)  # Prevent multiple pop-ups for the same document
            except:
                continue
        
        # Pause before checking again
        time.sleep(5)

if __name__ == "__main__":
    monitor_documents()
