import sys
import os

if getattr(sys, 'frozen', False):  # Check if running as an executable
    current_dir = os.path.dirname(sys.executable)  # Use the executable directory
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Use the script directory

sys.path.append(current_dir)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from wtforms import StringField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
# Add the path of AuMaintdb to sys.path
from modules.configuration.config import DATABASE_URL
from modules.configuration.base import Base


# Association table for tools and tool packages (many-to-many)

class ToolForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    size = StringField('Size')
    type = StringField('Type')
    material = StringField('Material')
    description = TextAreaField('Description')
    category = SelectField('Category', coerce=int, validators=[DataRequired()])
    manufacturer = SelectField('Manufacturer', coerce=int, validators=[DataRequired()])
    image_id = FileField('Image', validators=[FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('Add Tool')

# Database setup
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# Adding example data
def populate_example_data(session):
    # Check if "Hand Tools" already exists
    if not session.query(Category).filter_by(name="Hand Tools").first():
        # Create categories
        hand_tools = Category(name="Hand Tools", description="Manual tools operated by hand.")
        session.add(hand_tools)
        session.commit()

    # Check if manufacturers exist before adding
    if not session.query(Manufacturer).filter_by(name="Manufacturer A").first():
        manufacturer_a = Manufacturer(name="Manufacturer A", country="USA", website="https://www.manufacturera.com")
        session.add(manufacturer_a)

    if not session.query(Manufacturer).filter_by(name="Manufacturer B").first():
        manufacturer_b = Manufacturer(name="Manufacturer B", country="Germany", website="https://www.manufacturerb.de")
        session.add(manufacturer_b)

    # Similarly, add checks for other entries if necessary
    session.commit()

# Populate database with example data
populate_example_data(session)
print("Database populated with example data.")
