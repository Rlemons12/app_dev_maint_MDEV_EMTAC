import os
from flask import Flask, render_template
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from modules.emtacdb.emtacdb_fts import PowerPoint  # Replace with your actual database module

app = Flask(__name__)

# Define the path to the database directory and the database file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, 'Database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'emtac_db.db')

# Configure your database connection using the DATABASE_PATH variable
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/display_pdf/<int:powerpoint_id>')
def display_pdf(powerpoint_id):
    # Retrieve the PowerPoint record by ID
    powerpoint = session.query(PowerPoint).filter_by(id=powerpoint_id).first()
    
    if powerpoint:
        # Assuming the PDF file path is stored in the 'pdf_file_path' column
        pdf_file_path = powerpoint.pdf_file_path
        
        # Render a template to display the PDF file path
        return render_template('pdf_template.html', pdf_file_path=pdf_file_path)
    else:
        # Handle the case when PowerPoint with given ID is not found
        return "PowerPoint not found", 404

if __name__ == '__main__':
    app.run(debug=True)
