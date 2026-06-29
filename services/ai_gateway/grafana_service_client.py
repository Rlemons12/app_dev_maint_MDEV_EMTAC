from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------
ENV_PATH = Path(r"E:\emtac\dev_env\.env")
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
GRAFANA_ORG_ID = int(os.getenv("GRAFANA_ORG_ID", "1"))
GRAFANA_TIMEOUT = int(os.getenv("GRAFANA_TIMEOUT", "10"))
GRAFANA_MAX_DASHBOARDS = int(os.getenv("GRAFANA_MAX_DASHBOARDS", "5"))

GRAFANA_SYSTEM_PROMPT = os.getenv(
    "MCP_GRAFANA_SYSTEM_PROMPT",
    (
        "You are an observability assistant with access to Grafana context.\n"
        "Answer questions about dashboards, panels, alerts, and datasources.\n"
        "Be concise and specific. Reference panel/dashboard names when relevant.\n"
        "If asked to summarise alerts, list only firing or pending ones first.\n"
        "Do not greet the user.\n"
    ),
)

GRAFANA_KEYWORDS = {
    "grafana", "dashboard", "panel", "alert", "datasource",
    "prometheus", "loki", "tempo", "metric", "query", "visualization",
    "graph", "chart", "timeseries", "logs", "trace", "firing", "pending",
    "influxdb", "elasticsearch", "threshold",
}


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------
class GrafanaServiceClient:
    """
    Fetches observability context from a local Grafana instance
    to enrich GPU prompt requests.

    Public methods:
        health()                → bool
        get_dashboards()        → list[{uid, title, url, tags, folder}]
        get_dashboard_detail()  → {title, description, tags, panels}
        get_alerts()            → list[{name, state, severity, summary, starts_at}]
        get_datasources()       → list[{name, type, url, is_default}]
        build_context()         → formatted string ready for prompt injection
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        org_id: int = 1,
        timeout: int = 10,
        max_dashboards: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.timeout = timeout
        self.max_dashboards = max_dashboards
        self._headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Grafana-Org-Id": str(org_id),
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        logger.info(
            "GrafanaServiceClient initialized | base_url=%s | org_id=%s | max_dashboards=%s",
            self.base_url, self.org_id, self.max_dashboards,
        )

    # ---- internal --------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = requests.get(
            url, headers=self._headers, params=params or {}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    # ---- public ----------------------------------------------------------

    def health(self) -> bool:
        try:
            self._get("/api/health")
            return True
        except Exception as exc:
            logger.warning("Grafana health check failed: %s", exc)
            return False

    def get_dashboards(self, query: str = "") -> List[Dict[str, Any]]:
        """Return dashboards, optionally filtered by a search query."""
        try:
            params: Dict[str, Any] = {"type": "dash-db", "limit": self.max_dashboards}
            if query:
                params["query"] = query
            results = self._get("/api/search", params=params)
            return [
                {
                    "uid": d.get("uid", ""),
                    "title": d.get("title", ""),
                    "url": f"{self.base_url}{d.get('url', '')}",
                    "tags": d.get("tags", []),
                    "folder": d.get("folderTitle", "General"),
                }
                for d in (results if isinstance(results, list) else [])
            ]
        except Exception as exc:
            logger.warning("get_dashboards failed: %s", exc)
            return []

    def get_dashboard_detail(self, uid: str) -> Dict[str, Any]:
        """Return panels and meta for a specific dashboard uid."""
        try:
            data = self._get(f"/api/dashboards/uid/{uid}")
            dashboard = data.get("dashboard", {})
            panels = [
                {
                    "id": p.get("id"),
                    "title": p.get("title", ""),
                    "type": p.get("type", ""),
                    "description": p.get("description", ""),
                }
                for p in dashboard.get("panels", [])
            ]
            return {
                "title": dashboard.get("title", ""),
                "description": dashboard.get("description", ""),
                "tags": dashboard.get("tags", []),
                "panels": panels,
            }
        except Exception as exc:
            logger.warning("get_dashboard_detail(%s) failed: %s", uid, exc)
            return {}

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Return active alert summaries from Grafana unified alerting."""
        try:
            rules = self._get("/api/alertmanager/grafana/api/v2/alerts")
            alerts = []
            for rule in (rules if isinstance(rules, list) else []):
                labels = rule.get("labels", {})
                status = rule.get("status", {})
                alerts.append({
                    "name": labels.get("alertname", ""),
                    "state": status.get("state", ""),
                    "severity": labels.get("severity", ""),
                    "summary": rule.get("annotations", {}).get("summary", ""),
                    "starts_at": rule.get("startsAt", ""),
                })
            return alerts
        except Exception as exc:
            logger.warning("get_alerts failed: %s", exc)
            return []

    def get_datasources(self) -> List[Dict[str, Any]]:
        """Return all configured datasources."""
        try:
            results = self._get("/api/datasources")
            return [
                {
                    "name": ds.get("name", ""),
                    "type": ds.get("type", ""),
                    "url": ds.get("url", ""),
                    "is_default": ds.get("isDefault", False),
                }
                for ds in (results if isinstance(results, list) else [])
            ]
        except Exception as exc:
            logger.warning("get_datasources failed: %s", exc)
            return []

    def build_context(self, user_query: str = "") -> str:
        """
        Assemble a [GRAFANA CONTEXT] block for prompt injection.
        Uses user_query to filter dashboards for relevance.
        """
        sections: List[str] = ["[GRAFANA CONTEXT]"]

        datasources = self.get_datasources()
        if datasources:
            ds_lines = [
                f"  - {ds['name']} ({ds['type']})" + (" [default]" if ds["is_default"] else "")
                for ds in datasources
            ]
            sections.append("Datasources:\n" + "\n".join(ds_lines))

        dashboards = self.get_dashboards(query=user_query)
        if dashboards:
            dash_lines = []
            for d in dashboards:
                tags = ", ".join(d["tags"]) if d["tags"] else "none"
                dash_lines.append(
                    f"  - [{d['folder']}] {d['title']} (uid={d['uid']}, tags={tags})\n"
                    f"    {d['url']}"
                )
            sections.append("Dashboards:\n" + "\n".join(dash_lines))

            # Include panel detail when a single dashboard matched
            if len(dashboards) == 1:
                detail = self.get_dashboard_detail(dashboards[0]["uid"])
                if detail.get("panels"):
                    panel_lines = [
                        f"  - [{p['type']}] {p['title']}"
                        + (f": {p['description']}" if p.get("description") else "")
                        for p in detail["panels"][:10]
                    ]
                    sections.append(
                        f"Panels in '{detail['title']}':\n" + "\n".join(panel_lines)
                    )

        alerts = self.get_alerts()
        if alerts:
            active = [a for a in alerts if a.get("state") in ("firing", "pending")]
            if active:
                alert_lines = [
                    f"  - [{a['state'].upper()}] {a['name']}"
                    + (f" ({a['severity']})" if a.get("severity") else "")
                    + (f": {a['summary']}" if a.get("summary") else "")
                    for a in active[:10]
                ]
                sections.append("Active Alerts:\n" + "\n".join(alert_lines))
            else:
                sections.append("Active Alerts: none")

        return "\n\n".join(sections) + "\n[END GRAFANA CONTEXT]"


# ---------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------
grafana_service_client = GrafanaServiceClient(
    base_url=GRAFANA_URL,
    api_key=GRAFANA_API_KEY,
    org_id=GRAFANA_ORG_ID,
    timeout=GRAFANA_TIMEOUT,
    max_dashboards=GRAFANA_MAX_DASHBOARDS,
)