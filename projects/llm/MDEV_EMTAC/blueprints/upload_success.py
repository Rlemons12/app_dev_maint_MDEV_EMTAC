# blueprints/upload_success.py

from flask import Flask, render_template


app = Flask(__name__)


@app.route('/success')
def upload_success():
    return render_template('success.html')
