#!/usr/bin/env python
"""
Render deployment startup script
Ensures proper port binding for cloud deployment
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment (Render sets this) or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("APP_HOST", "0.0.0.0")

    print(f"Starting server on {host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
    )
