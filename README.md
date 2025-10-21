# 🧠 Smart Data Labeler – AI-Assisted Text Annotation Tool

**Semi-automated text labeling made simple** 🚀  
Built with **Python Flask + Pandas + HTML/CSS**, this tool allows you to:

- Upload CSV datasets ✅  
- Get AI-like label suggestions (mock AI) 💡  
- Manually label and save text ✅  
- Export labeled data for ML workflows ✅  

This project is perfect for **portfolio showcases, resume projects, and practical GenAI workflows**.

---

## ⚡ Quick Demo

1. Upload a CSV file with a `text` column  
2. View each text entry on the labeling page  
3. Enter labels manually or click **💡 Get AI Suggestion**  
4. Save results → exported automatically as `labeled_data.csv`  

---

## 🧩 Features

- **Upload CSV datasets** – easy and fast  
- **Mock AI suggestions** – provides label recommendations  
- **Manual labeling** – full control over labeling  
- **Export labeled data** – ready for ML pipelines  
- **Portfolio & Resume Ready** – no paid API required  

---

## 📁 Folder Structure

smart-data-labeler/
│
├── app.py
├── requirements.txt
├── templates/
│ ├── index.html
│ ├── label.html
├── static/
│ └── style.css
├── data/
│ ├── sample.csv
│ ├── uploaded.csv
│ └── labeled_data.csv
└── README.md
---

## ⚙️ Tech Stack

| Layer    | Technology          |
|----------|-------------------|
| Backend  | Python (Flask)     |
| Frontend | HTML, CSS, Jinja2  |
| Data     | CSV files           |
| AI       | Mock AI suggestion  |
| Others   | Pandas for data handling |

---

## 🛠 Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-data-labeler.git
cd smart-data-labeler
Create and activate virtual environment

bash
Copy code
python -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # Mac/Linux
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
python app.py
Open in browser → http://127.0.0.1:5000

🌟 Future Enhancements
Multi-annotator login system

Dataset statistics & progress tracking

Integration with real AI APIs (OpenAI, Hugging Face)

Support for multi-label classification

👨‍💻 Author
C Keshav Reddy
💼 Aspiring GenAI & Data Engineer | Innovation-focused Builder

🔗 LinkedIn : 

“Label smarter, not harder.” ⚡
