"""Compatibility launcher for the canonical FastAPI application.

This keeps legacy commands like `python app.py` and `gunicorn app:app`
working while routing all traffic through `main.py`.
"""

import os
import uvicorn
from main import app

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
    # uvicorn.run("main:app", host=host, port=port, reload=True)