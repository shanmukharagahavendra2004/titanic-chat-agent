import os
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class TitanicAgent:
    def __init__(self, csv_path: str):
        if not csv_path or not csv_path.strip():
            raise ValueError("CSV path cannot be empty.")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at '{csv_path}'. Please check the file path.")

        try:
            self.df = pd.read_csv(csv_path)
            if self.df.empty:
                raise ValueError("The dataset is empty. Please provide a valid Titanic CSV file.")
            logger.info(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except pd.errors.ParserError as e:
            raise RuntimeError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading dataset: {e}")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found. Please add it to your .env file.")

        try:
            self.llm = ChatGroq(
                api_key=api_key,
                model="llama-3.3-70b-versatile",
                temperature=0,
            )
            logger.info("Groq LLaMA 3 LLM initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

        try:
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type="tool-calling",
            )
            logger.info("Pandas DataFrame Agent created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create pandas agent: {e}")

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return "Please provide a valid question."

        try:
            response = self.agent.invoke({"input": question})
            answer = response.get("output", "").strip()
            if not answer:
                return "I could not find an answer. Please try rephrasing your question."
            return answer
        except ValueError as e:
            logger.error(f"Invalid input to agent: {e}")
            return "Your question could not be understood. Please try rephrasing it."
        except Exception as e:
            logger.error(f"Agent error for question '{question}': {e}")
            return "Something went wrong while processing your question. Please try again."