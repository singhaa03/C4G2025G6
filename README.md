# C4G2025G6
SPAM DETECTION AI EXTENSION

Setup Instructions:
1. Clone the repository
2. git clone https://github.com/singhaa03/C4G2025G6.git
cd C4G2025G6
3. Set up Python environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
4. Download the trained model
bashpython download_model.py
This will automatically download and extract the model files to the model/ folder.
5. Start the backend server
bashcd backend
uvicorn main:app --reload
6. Load the browser extension

Open Chrome and go to chrome://extensions/
Enable "Developer mode"
Click "Load unpacked"
Select the extension folder from this project

6. Test the application

The API will be running at http://127.0.0.1:8000
Use the browser extension to test phishing detection

Project Structure
C4G2025G6/
├── backend/
│   ├── main.py          # FastAPI server
│   └── requirements.txt
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── styles.css
├── model/               # Downloaded model files (auto-created)
├── download_model.py    # Script to download trained model
└── train.py            # Model training script
Troubleshooting

If model download fails, check your internet connection
Make sure you have Python 3.7+ installed
For extension issues, check the browser console for errors
