import streamlit as st
from newspaper import Article
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import textwrap
import PyPDF2
import tempfile

# --------- Config ---------
MODELS = {
    "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
    "T5 Base (t5-base)": "t5-base",
    "Pegasus (google/pegasus-xsum)": "google/pegasus-xsum",
    "Falcon (Falconsai/text_summarization)": "Falconsai/text_summarization",
    "Meeting Summary (knkarthick/MEETING_SUMMARY)": "knkarthick/MEETING_SUMMARY"
}
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Load Summarization Model ---------
@st.cache_resource
def load_summarizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

# --------- Load Chat Model ---------
@st.cache_resource
def load_chat_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(DEVICE)
    return tokenizer, model

# --------- Summarizer ---------
def summarize_long_text(text, tokenizer, model):
    chunks = textwrap.wrap(text, CHUNK_SIZE)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE).to(DEVICE)
        summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return "\n\n".join(summaries)

# --------- File Handling ---------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --------- Chat Interface ---------
def chat_with_summary(summary_text):
    chat_tokenizer, chat_model = load_chat_model()
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None

    st.subheader("💬 Chat with the Summary")
    user_input = st.text_input("Ask something about the article:")
    if user_input:
        new_input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors="pt").to(DEVICE)
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

        st.session_state.chat_history_ids = chat_model.generate(bot_input_ids, max_length=1000, pad_token_id=chat_tokenizer.eos_token_id)
        response = chat_tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        st.markdown(f"**Bot:** {response}")

# --------- UI ---------
st.set_page_config(page_title="🧠 Smart Summarizer", layout="centered")
st.title("🧠 Smart Summarizer & Chat")
st.markdown("Summarize any **URL**, **PDF**, or **Text**, then chat with the result using open-source models.")

option = st.radio("Choose input type:", ["URL", "Upload PDF/Text"])
model_label = st.selectbox("Choose summarization model", list(MODELS.keys()))
model_name = MODELS[model_label]

text = ""

if option == "URL":
    url = st.text_input("Enter a URL:")
    if st.button("Fetch & Summarize"):
        if url:
            with st.spinner("Downloading article..."):
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    text = article.text
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a URL.")

elif option == "Upload PDF/Text":
    uploaded_file = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])
    if uploaded_file and st.button("Summarize File"):
        with st.spinner("Extracting and summarizing..."):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    text = uploaded_file.read().decode("utf-8")
                else:
                    st.error("Unsupported file type.")
            except Exception as e:
                st.error(f"Error: {e}")

if text:
    with st.spinner("Summarizing text..."):
        tokenizer, model = load_summarizer(model_name)
        summary = summarize_long_text(text, tokenizer, model)
        st.subheader("📝 Summary")
        st.success(summary)
        chat_with_summary(summary)
