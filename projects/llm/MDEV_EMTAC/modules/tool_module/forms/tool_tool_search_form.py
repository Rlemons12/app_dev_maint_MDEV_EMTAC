# modules/tool_module/forms/tool_search.py

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, SelectMultipleField
from wtforms.validators import Optional
from wtforms.widgets import ListWidget, CheckboxInput

class ToolSearchForm(FlaskForm):
    tool_name = StringField('Tool Name', validators=[Optional()])
    tool_size = StringField('Tool Size', validators=[Optional()])
    tool_type = StringField('Tool Type', validators=[Optional()])
    tool_material = StringField('Tool Material', validators=[Optional()])

    # If you want to allow multiple categories/manufacturers to be selected
    tool_category = SelectMultipleField(
        'Tool Category',
        coerce=int,
        validators=[Optional()],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    tool_manufacturer = SelectMultipleField(
        'Tool Manufacturer',
        coerce=int,
        validators=[Optional()],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )

    submit_search = SubmitField('Search')

