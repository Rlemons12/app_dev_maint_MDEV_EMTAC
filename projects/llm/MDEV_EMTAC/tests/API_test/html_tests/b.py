from flask import Flask, render_template
import webbrowser
import threading
import time
import os

# Get the current directory of this file
current_dir = os.path.dirname(__file__)

# Go up three directories: html_tests -> API_test -> Tests -> AuMaintdb
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# Set the template_folder to the 'templates' folder at the AuMaintdb level
app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'))

@app.route('/')
def home():
    # Template is at AuMaintdb/templates/module_template_html/test_template.html
    return render_template('module_template_html/test_template.html')

def open_browser(port):
    # Wait a bit to ensure the server starts before opening the browser
    time.sleep(1)
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == '__main__':
    port = 5001
    # Start a thread to open the browser after a short delay
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Run the app on port 5001
    app.run(debug=True, port=port)
