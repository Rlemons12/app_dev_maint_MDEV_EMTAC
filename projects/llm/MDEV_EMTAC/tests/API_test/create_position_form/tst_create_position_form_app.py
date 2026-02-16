import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash

# --------------------------------------------------------------
# Fix Python path
# --------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger
from modules.emtacdb.forms.tst_create_position_form import (
    CreatePositionForm,
    cnp_form_create_position
)
from modules.emtacdb.emtacdb_fts import Position


# --------------------------------------------------------------
# Flask App
# --------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "test123"

# Global DB instance
db = DatabaseConfig()


@app.route("/test/create-position", methods=["GET", "POST"])
def test_create_position():
    session = db.get_main_session()

    form = CreatePositionForm()
    form.set_query_factories(session)

    if request.method == "POST":
        try:
            pos_id = form.save(
                session=session,
                cnp_form_create_position=cnp_form_create_position
            )
            flash(f"Position created or reused ID={pos_id}", "success")
            return redirect(url_for("test_create_position"))
        except Exception as e:
            logger.exception("Error in POST /test/create-position")
            flash(str(e), "danger")

    positions = session.query(Position).order_by(Position.id.desc()).limit(15).all()
    return render_template("test_create_position_form.html", form=form, positions=positions)


if __name__ == "__main__":
    app.run(debug=True)
