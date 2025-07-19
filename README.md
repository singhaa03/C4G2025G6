# C4G2025G6
SPAM DETECTION AI EXTENSION
* Type everything inside the quotations,NOT the quotations itself.
Setup Instructions:
1. To Clone the repository do this -> 
2. Go to mac terminal then type, "git clone https://github.com/singhaa03/C4G2025G6.git"
3. Then download vs code and type, cd C4G2025G6
4. Set up Python environment:
  - python -m venv venv
  - source venv/bin/activate  # On Windows: venv\Scripts\activate
  - pip install fastapi uvicorn transformers torch
* THIS IS ALL YOU HAVE TO IMPORT, BUT IF There is a import error ask ChatGPT 
5. Download the trained model by typing:
  - "python download_model.py"
This will automatically download and extract the model files to the model/ folder.
6. Start the backend server by typing :
  - "cd backend" then,
  - "uvicorn main:app --reload"
7. Load the browser extension by :
  - Open Chrome and go to chrome://extensions/
  - Enable "Developer mode"
  - Click "Load unpacked"
  - Go to C4G2025G6 folder
  - Select the extension folder from this project

6. Test the application
  - CLick on hte extensions Icon, and test it out!

Structure(if needed): 
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
