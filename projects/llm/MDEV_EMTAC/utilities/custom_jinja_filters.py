# utilities/custom_jinja_filters.py

from flask import Flask


def register_jinja_filters(app):
    """
    Register custom Jinja2 filters for the application.

    Args:
        app (Flask): The Flask application instance
    """

    @app.template_filter('attr_names')
    def attr_names_filter(obj):
        """
        Get all attribute names of an object.

        Args:
            obj: The object to inspect

        Returns:
            list: List of attribute names
        """
        try:
            return [attr for attr in dir(obj) if not attr.startswith('_')]
        except Exception:
            return []

    @app.template_filter('has_attr')
    def has_attr_filter(obj, attr_name):
        """
        Check if an object has a specific attribute.

        Args:
            obj: The object to inspect
            attr_name (str): The attribute name to check

        Returns:
            bool: True if the object has the attribute, False otherwise
        """
        try:
            return hasattr(obj, attr_name)
        except Exception:
            return False

    # Add the filter registration to your app initialization code
    # Example usage in app.py or __init__.py:
    #
    # from flask import Flask
    # app = Flask(__name__)
    # register_jinja_filters(app)
    #
    # Or if using blueprints and application factory pattern:
    #
    # def create_app():
    #     app = Flask(__name__)
    #     register_jinja_filters(app)
    #     return app