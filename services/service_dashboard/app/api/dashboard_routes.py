from __future__ import annotations

from flask import Blueprint, render_template

from configuration.config import GRAFANA_URL


dashboard_bp = Blueprint("dashboard_bp", __name__)


def _build_dashboard_context() -> dict[str, str]:
    """
    Build template context for dashboard pages.

    Notes:
    - Use relative URLs for routes served by this Flask app so they work
      regardless of hostname or IP address.
    - Use configured absolute URL for Grafana because it is a separate service.
    """
    return {
        "grafana_url": GRAFANA_URL,
        "trace_url": "/dashboards/trace",
        "services_url": "/dashboards/services",
        "gpu_url": "/dashboards/gpu",
        "ai_url": "/dashboards/ai",
        "grafana_dashboard_url": "/dashboards/grafana",
    }


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboards/services")
def dashboard_services():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/gpu")
def dashboard_gpu():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/ai")
def dashboard_ai():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/trace")
def dashboard_trace():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/grafana")
def dashboard_grafana():
    """
    Keeps a dashboard route for Grafana-related navigation inside the app.
    The actual Grafana UI still lives at GRAFANA_URL.
    """
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)