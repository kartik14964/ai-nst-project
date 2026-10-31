import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nst_code_dir = os.path.join(BASE_DIR, 'NST_Code')
if nst_code_dir not in sys.path:
    sys.path.insert(0, nst_code_dir)

from app import app

if __name__ == '__main__':
    print("Starting AdaIN Neural Style Transfer Web App...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)

