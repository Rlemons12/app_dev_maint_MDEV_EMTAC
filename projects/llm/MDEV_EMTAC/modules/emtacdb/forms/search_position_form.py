# modules/position_module/forms/search_position_form.py

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, DateField
from wtforms.validators import Optional

class SearchPositionForm(FlaskForm):
    position_id = StringField('Position ID', validators=[Optional()])
    position_name = StringField('Position Name', validators=[Optional()])
    area = SelectField('Area', coerce=int, validators=[Optional()])
    equipment_group = SelectField('Equipment Group', coerce=int, validators=[Optional()])
    model = SelectField('Model', coerce=int, validators=[Optional()])
    start_date = DateField('Start Date', format='%Y-%m-%d', validators=[Optional()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[Optional()])
    search = SubmitField('Search')
