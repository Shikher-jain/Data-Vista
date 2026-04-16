"""Backward-compatible FastAPI module entrypoint.

Use `main.py` as the single source of truth.
"""

import os

import uvicorn

from main import app


if __name__ == "__main__":
	host = os.getenv("HOST", "127.0.0.1")
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host=host, port=port, reload=False)