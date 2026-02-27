# 🚢 Titanic Dataset Chatbot

A conversational AI chatbot that lets you ask plain-English questions about the Titanic passenger dataset and get back intelligent text answers along with beautiful visualizations — all inside a clean Streamlit interface.

---

## 📸 Features

- 💬 Ask natural language questions about the Titanic dataset
- 📊 Auto-generates charts and histograms based on your question
- 🤖 Powered by LLaMA 3 (via Groq) and LangChain agents
- ⚡ FastAPI backend with Streamlit frontend
- 🧠 Pandas DataFrame Agent for accurate data analysis

---

## 🏗️ Project Structure

```
titanic-chatbot/
├── api.py               # FastAPI backend — agent + chart generation
├── app.py               # Streamlit frontend — chat UI
├── titanic.csv          # Titanic dataset
├── .env                 # API keys (not committed to git)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Agent Framework | LangChain |
| LLM | LLaMA 3.3 70B via Groq |
| Data Analysis | Pandas |
| Visualization | Matplotlib + Seaborn |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/titanic-chatbot.git
cd titanic-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [https://console.groq.com](https://console.groq.com)

---

## 🚀 Running the App

You need **two terminals** running simultaneously.

### Terminal 1 — Start the FastAPI backend

```bash
uvicorn api:app --reload
```

Backend will be live at: `http://127.0.0.1:8000`

### Terminal 2 — Start the Streamlit frontend

```bash
streamlit run app.py
```

Frontend will open at: `http://localhost:8501`

---

## 💬 Example Questions

| Question | Response Type |
|---|---|
| What percentage of passengers were male? | Text + Pie chart |
| Show me a histogram of passenger ages | Histogram |
| What was the average ticket fare? | Text answer |
| How many passengers embarked from each port? | Bar chart |
| Show me a survival chart | Bar chart |
| Show passenger class distribution | Bar chart |
| What was the survival rate? | Text answer |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat` | Send a question, get answer + optional chart |

### POST `/chat` — Request body

```json
{
  "question": "Show me a histogram of passenger ages"
}
```

### POST `/chat` — Response

```json
{
  "success": true,
  "question": "Show me a histogram of passenger ages",
  "answer": "Here is the chart for you! 📊",
  "chart": "<base64 encoded PNG string or null>"
}
```

---

## 📦 Dataset

The Titanic dataset (`titanic.csv`) contains information about **891 passengers** with the following key columns:

| Column | Description |
|---|---|
| `Survived` | 0 = No, 1 = Yes |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Male / Female |
| `Age` | Passenger age |
| `Fare` | Ticket price |
| `Embarked` | Port (S = Southampton, C = Cherbourg, Q = Queenstown) |

> Download the dataset from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)

---

## 🔧 Configuration

You can swap the LLM model in `api.py`:

```python
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",  # Change model here
    temperature=0,
)
```

Available Groq models:

| Model | Speed | Quality |
|---|---|---|
| `llama-3.3-70b-versatile` | Medium | Best |
| `llama-3.1-8b-instant` | Fastest | Good |
| `mixtral-8x7b-32768` | Medium | Very Good |

---
