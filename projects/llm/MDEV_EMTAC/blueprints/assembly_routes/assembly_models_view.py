from flask import render_template, request, flash
from . import assembly_model_bp
from flask import render_template

@assembly_model_bp.route('/assembly_model', methods=['GET'])
def assembly_model_page():
    """
    Route to render the assembly_model.html template.
    """
    return render_template('assembly_model.html')


@assembly_model_bp.route('/new', methods=['GET', 'POST'])
def new_assembly():
    if request.method == 'POST':
        # Handle form submission logic here
        flash("New assembly created successfully!", "success")
    return render_template('assembly_model/new_assembly.html')
