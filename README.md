User‑Response‑Prediction‑System
A Flask‑based web app that predicts whether a user will click on an online advertisement (CTR prediction) using machine‑learning models.


Wireframe & Documentation:

Project Documentation (PDF)

Wireframe (PDF)

Dataset: Dataset/advertising.csv

🛠️ Built With
Backend & Web: Flask

Machine Learning: scikit‑learn (Logistic Regression, Random Forest, XGBoost, SVM, KNN)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Deployment: Heroku

⚙️ Features
User inputs:

Daily Time Spent on Site

Age

Area Income

Daily Internet Usage

Gender (1 = Male, 0 = Female)

Predicts probability of “Clicked on Ad” (0 = No, 1 = Yes)

Clean, responsive UI with HTML/CSS/JavaScript

Model pipeline with feature scaling and serialized via pickle

🏁 Getting Started
Clone the repo

bash
Copy
Edit
git clone https://github.com/7Vivek/User-Response-Prediction-System.git
cd User-Response-Prediction-System
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
python app.py
Open your browser at http://localhost:5000

📂 Project Structure
bash
Copy
Edit
├── app.py                 # Flask application
├── model.pkl              # Trained ML pipeline
├── requirements.txt       # Python dependencies
├── Dataset/
│   └── advertising.csv    # Source data
├── templates/
│   └── index.html         # Front‑end form
├── static/
│   ├── css/style.css      # Styles
│   └── js/                # Scripts (anime.js, etc.)
└── Documentation/
    ├── *.pdf              # Project docs & wireframe
📈 Experimental Results
Model	Accuracy
Logistic Regression	95.3%
Random Forest Classifier	96.0%
XGBClassifier	95.0%
Linear SVC	95.3%
K‑Nearest Neighbors	95.0%

🤝 Contributing
Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you’d like to change.


