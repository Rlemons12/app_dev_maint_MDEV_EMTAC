import threading
import time
import webbrowser
from flask import render_template, redirect, url_for
from modules.tool_module.model import ToolCategory, Manufacturer, ToolForm
from modules.tool_module import create_app,db
from modules.emtacdb.emtacdb_fts import Image, Tool

# Create Flask app using the factory function
app = create_app()



@app.route('/')
def home():
    return redirect(url_for('add_tool'))  # Redirect to /add_tool

@app.route('/add_tool', methods=['GET', 'POST'])
def add_tool():
    form = ToolForm()

    # Load available choices for categories, manufacturers, and images
    form.category.choices = [(c.id, c.name) for c in db.session.query(ToolCategory).all()]
    form.manufacturer.choices = [(m.id, m.name) for m in db.session.query(Manufacturer).all()]
    form.image_ids.choices = [(i.id, i.title) for i in db.session.query(Image).all()]  # Populate images

    if form.validate_on_submit():
        # Add a new tool to the database if form is submitted
        new_tool = Tool(
            name=form.name.data,
            size=form.size.data,
            type=form.type.data,
            material=form.material.data,
            description=form.description.data,
            category_id=form.category.data,
            manufacturer_id=form.manufacturer.data
        )
        db.session.add(new_tool)

        # Associate selected images with the tool
        for image_id in form.image_ids.data:
            image = db.session.query(Image).get(image_id)
            new_tool.images.append(image)

        db.session.commit()
        return redirect(url_for('add_tool'))

    return render_template('add_tool.html', form=form)


def open_browser():
    """Open the default web browser after a delay to give the server time to start"""
    time.sleep(1)  # Delay to ensure the server has started
    webbrowser.open_new('http://127.0.0.1:5000/add_tool')


if __name__ == '__main__':
    # Start a thread to open the web browser
    threading.Thread(target=open_browser).start()

    # Run the Flask application
    app.run(debug=True, port=5000)
