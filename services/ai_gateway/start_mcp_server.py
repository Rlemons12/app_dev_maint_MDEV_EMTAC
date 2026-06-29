from __future__ import annotations

import logging
import sys
from pathlib import Path

import uvicorn

# ---------------------------------------------------------
# Bootstrap project root so "services.*" imports work
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
AI_GATEWAY_DIR = CURRENT_FILE.parent                      # ...\services\ai_gateway
SERVICES_DIR = AI_GATEWAY_DIR.parent                      # ...\services
PROJECT_ROOT = SERVICES_DIR.parent                        # ...\emtac

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting EMTAC MCP Gateway")
    logger.info("CURRENT_FILE=%s", CURRENT_FILE)
    logger.info("PROJECT_ROOT=%s", PROJECT_ROOT)

    uvicorn.run(
        "services.ai_gateway.mcp_server:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()