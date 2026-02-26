import os
import io
import base64
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Please add it to your .env file.")

app = FastAPI(title="Titanic Chatbot API 🚢")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "An unexpected server error occurred. Please try again."}
    )

CSV_PATH = "titanic.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found at '{CSV_PATH}'. Please place titanic.csv in the project root.")

try:
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")

try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
    logger.info("Groq LLM initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

SYSTEM_PROMPT = """You are a helpful data analyst assistant for the Titanic dataset.
Answer questions directly with facts and numbers from the dataset.
NEVER write or explain Python code. NEVER describe how to make a chart.
If a visualization is requested, simply say: 'Here is the chart for you!'
Always give short, clear, data-driven answers."""

try:
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        prefix=SYSTEM_PROMPT,
    )
    logger.info("LangChain agent initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to create pandas agent: {e}")


def fig_to_base64(fig) -> str:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return encoded
    except Exception as e:
        logger.error(f"Chart encoding failed: {e}")
        return None
    finally:
        plt.close(fig)


def try_generate_chart(question: str):
    q = question.lower()
    try:
        if ("histogram" in q or "distribution" in q or "hist" in q) and "age" in q:
            fig, ax = plt.subplots()
            ax.hist(df["Age"].dropna(), bins=20, color="steelblue", edgecolor="white")
            ax.set_title("Age Distribution of Passengers")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            return fig_to_base64(fig)

        if ("histogram" in q or "distribution" in q or "hist" in q) and "fare" in q:
            fig, ax = plt.subplots()
            ax.hist(df["Fare"].dropna(), bins=30, color="coral", edgecolor="white")
            ax.set_title("Fare Distribution")
            ax.set_xlabel("Fare")
            ax.set_ylabel("Count")
            return fig_to_base64(fig)

        if "embarked" in q or "port" in q or "embark" in q:
            fig, ax = plt.subplots()
            counts = df["Embarked"].value_counts()
            counts.index = counts.index.map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
            ax.set_title("Passengers by Embarkation Port")
            ax.set_xlabel("Port")
            ax.set_ylabel("Number of Passengers")
            return fig_to_base64(fig)

        if "surviv" in q and any(w in q for w in ["chart", "bar", "show", "plot", "visuali"]):
            fig, ax = plt.subplots()
            counts = df["Survived"].value_counts().rename({0: "Did Not Survive", 1: "Survived"})
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="RdYlGn")
            ax.set_title("Survival Count")
            ax.set_xlabel("Outcome")
            ax.set_ylabel("Number of Passengers")
            return fig_to_base64(fig)

        if any(w in q for w in ["male", "female", "gender", "sex"]) and \
           any(w in q for w in ["chart", "pie", "show", "plot", "visuali", "percentage", "%"]):
            fig, ax = plt.subplots()
            counts = df["Sex"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["steelblue", "coral"])
            ax.set_title("Gender Distribution")
            return fig_to_base64(fig)

        if "class" in q and any(w in q for w in ["chart", "bar", "show", "plot", "visuali"]):
            fig, ax = plt.subplots()
            counts = df["Pclass"].value_counts().sort_index()
            counts.index = ["1st Class", "2nd Class", "3rd Class"]
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
            ax.set_title("Passengers by Class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            return fig_to_base64(fig)

    except Exception as e:
        logger.error(f"Chart generation failed for question '{question}': {e}")
        return None

    return None


def is_chart_only_question(question: str) -> bool:
    q = question.lower()
    chart_words = ["histogram", "hist", "bar chart", "pie chart", "plot", "show me", "visuali"]
    data_words = ["age", "fare", "embarked", "port", "embark", "survived", "survival",
                  "male", "female", "gender", "sex", "class"]
    return any(w in q for w in chart_words) and any(w in q for w in data_words)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Titanic Chatbot Backend is Running 🚢"}


@app.post("/chat")
def chat(request: QueryRequest):
    if not request.question or not request.question.strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Question cannot be empty."}
        )

    try:
        chart_base64 = try_generate_chart(request.question)

        if chart_base64 and is_chart_only_question(request.question):
            return {
                "success": True,
                "question": request.question,
                "answer": "Here is the chart for you! 📊",
                "chart": chart_base64
            }

        try:
            response = agent.invoke({"input": request.question})
            answer = response.get("output", "").strip()
            if not answer:
                answer = "I could not find an answer for that. Please try rephrasing your question."
        except Exception as agent_error:
            logger.error(f"Agent failed: {agent_error}")
            answer = "The AI agent encountered an issue processing your question. Please try again."

        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "chart": chart_base64
        }

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Something went wrong. Please try again later."}
        )