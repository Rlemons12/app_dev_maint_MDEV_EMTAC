from app.main import app
from configuration.config import SERVICE_DASHBOARD_HOST, SERVICE_DASHBOARD_PORT


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        host=SERVICE_DASHBOARD_HOST,
        port=SERVICE_DASHBOARD_PORT,
    )