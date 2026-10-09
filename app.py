import logging
import uvicorn
from ekis.api import app, logger

if __name__ == "__main__":
    logger.info("Starting API server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
