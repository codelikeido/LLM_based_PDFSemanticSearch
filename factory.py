"""
factory.py - Flask application factory.

Usage:
  from app.factory import create_app
  app = create_app()
  app.run()
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask

from app.config import DEBUG, MAX_CONTENT_MB, SECRET_KEY


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Core config ───────────────────────────────────────────────────────────
    app.config.update(
        SECRET_KEY          = SECRET_KEY,
        MAX_CONTENT_LENGTH  = MAX_CONTENT_MB * 1024 * 1024,
        JSON_SORT_KEYS      = False,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
        stream  = sys.stdout,
        level   = log_level,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ── Register blueprints ───────────────────────────────────────────────────
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    # ── Root route ────────────────────────────────────────────────────────────
    @app.route("/")
    def root():
        from flask import jsonify
        return jsonify({
            "service": "PDF Semantic Search API",
            "version": "1.0.0",
            "docs":    "/api/v1/health",
        })

    return app
