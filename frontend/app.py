import os
import streamlit as st
import requests
import base64
import io
from PIL import Image, UnidentifiedImageError

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL","")

st.set_page_config(page_title="Titanic Chatbot 🚢", layout="centered")
st.title("🚢 Titanic Dataset Chatbot")
st.write("Ask me anything about the Titanic dataset!")

if "messages" not in st.session_state:
    st.session_state.messages = []


def decode_chart(chart_base64: str):
    try:
        if not chart_base64:
            return None
        img_bytes = base64.b64decode(chart_base64)
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        st.warning(f"Chart could not be rendered: {e}")
        return None
    except Exception as e:
        st.warning(f"Unexpected error rendering chart: {e}")
        return None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart"):
            img = decode_chart(message["chart"])
            if img:
                st.image(img, width="stretch")


if prompt := st.chat_input("Ask a question about Titanic dataset..."):
    if not prompt.strip():
        st.warning("Please enter a valid question.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        bot_reply = None
        chart_base64 = None

        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"question": prompt},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

                if data.get("success"):
                    bot_reply = data.get("answer", "No answer returned.")
                    chart_base64 = data.get("chart")
                else:
                    bot_reply = f"⚠️ {data.get('error', 'Unknown error from server.')}"

            except requests.exceptions.ConnectionError:
                bot_reply = "❌ Cannot connect to the backend. Make sure the FastAPI server is running."
            except requests.exceptions.Timeout:
                bot_reply = "⏱️ The request timed out. The server is taking too long to respond. Please try again."
            except requests.exceptions.HTTPError as e:
                bot_reply = f"🚫 Server returned an error: {e.response.status_code} - {e.response.reason}"
            except ValueError:
                bot_reply = "⚠️ Received an invalid response from the server. Please try again."
            except Exception as e:
                bot_reply = f"❌ Unexpected error: {str(e)}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_reply,
            "chart": chart_base64
        })

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            if chart_base64:
                img = decode_chart(chart_base64)
                if img:
                    st.image(img, width="stretch")