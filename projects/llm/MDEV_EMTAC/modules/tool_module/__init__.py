import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    template_dir = os.path.join(os.path.dirname(__file__), 'templates/tools')
    print(f"Resolved template directory: {template_dir}")
    app = Flask(
        __name__,
        template_folder=template_dir
    )

    # Flask app configurations
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tools_inventory.db'
    app.config['SECRET_KEY'] = '12345'

    # Initialize database
    db.init_app(app)

    with app.app_context():
        from .model.tool_model import Base
        Base.metadata.create_all(db.engine)

    return app

__all__ = ['create_app', 'db']
