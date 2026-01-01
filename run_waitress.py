from waitress import serve
import app  # This should be the Python file where your Flask app is defined

# Assuming in app.py you have:
# app = Flask(__name__)

serve(app.app, host='0.0.0.0', port=5000)
