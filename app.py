"""
Root launcher for running the Flask Frontend locally.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting Local NST Web App on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
