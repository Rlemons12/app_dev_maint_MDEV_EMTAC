# manufacturer_form.py

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.fields import TextAreaField
from wtforms.validators import DataRequired, Optional, URL, ValidationError
from modules.emtacdb.emtacdb_fts import ToolManufacturer

class ToolManufacturerForm(FlaskForm):
    name = StringField('Manufacturer Name', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[Optional()])
    country = StringField('Country', validators=[Optional()])
    website = StringField('Website URL', validators=[Optional(), URL()])
    submit = SubmitField('Submit Manufacturer')

    '''def validate_name(self, name):
        manufacturer = ToolManufacturer.query.filter_by(name=name.data.strip()).first()
        if manufacturer:
            raise ValidationError('This manufacturer name is already in use. Please choose a different name.')
'''