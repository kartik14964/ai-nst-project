import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NST_DIR  = os.path.join(BASE_DIR, 'NST_Code')
if NST_DIR not in sys.path:
    sys.path.insert(0, NST_DIR)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Neural Style Transfer on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
