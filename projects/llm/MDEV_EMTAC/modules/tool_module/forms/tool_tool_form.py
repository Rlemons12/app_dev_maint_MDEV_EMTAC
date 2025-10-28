# modules/tool_module/forms/tool_add.py

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, FileField, SubmitField, SelectMultipleField
from wtforms.validators import DataRequired, Length, Optional
from flask_wtf.file import MultipleFileField, FileAllowed,FileRequired  # If you plan to use these validators


class ToolForm(FlaskForm):
    tool_name = StringField('Tool Name', validators=[DataRequired()])
    tool_size = StringField('Tool Size', validators=[Optional()])
    tool_type = StringField('Tool Type', validators=[Optional()])
    tool_material = StringField('Tool Material', validators=[Optional()])
    tool_description = TextAreaField('Tool Description', validators=[Optional()])
    tool_category = SelectField('Tool Category', coerce=int, validators=[DataRequired()])
    tool_manufacturer = SelectField('Tool Manufacturer', coerce=int, validators=[DataRequired()])
    tool_images = MultipleFileField('Tool Images', validators=[
        Optional(), FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    image_description = TextAreaField('Image Description', validators=[Optional()])
    submit_tool = SubmitField('Submit Tool Data')