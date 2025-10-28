# tool_category_forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Optional, ValidationError
from modules.emtacdb.emtacdb_fts import ToolCategory  # Adjust the import path
from modules.configuration.config_env import DatabaseConfig

db_config = DatabaseConfig

class ToolCategoryForm(FlaskForm):
    name = StringField('Category Name', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[Optional()])
    parent_id = SelectField('Parent Category', coerce=int, validators=[Optional()])
    submit = SubmitField('Add Category')

    """def validate_name(self, name):
        
        Custom validator to ensure the category name is unique.
        
        category = ToolCategory.query.filter_by(name=name.data.strip()).first()
        if category:
            raise ValidationError('This category name is already in use. Please choose a different name.')
"""