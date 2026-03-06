import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main():

    logging.info("Starting EMTAC MCP Gateway")

    uvicorn.run(
        "services.ai_gateway.mcp_server:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()