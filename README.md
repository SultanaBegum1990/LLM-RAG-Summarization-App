# 📝 Webpage Summarizer App

A **Streamlit-based web app** that extracts and summarizes content from webpages or custom text using **Hugging Face Transformers** models.

---

## 🔍 Features

- 🌐 Summarize any article URL.
- ✍️ Summarize user-pasted text.
- 🧠 Choose from popular Hugging Face models (BART, T5, Pegasus, LLaMA, etc.).
- ⚡ GPU-compatible (if available).

---

## 📦 Supported Models

| Display Name                 | Hugging Face Model ID              |
|-----------------------------|------------------------------------|
| BART                        | `facebook/bart-large-cnn`         |
| T5 Base                     | `t5-base`                          |
| Pegasus XSum                | `google/pegasus-xsum`             |
| Falcon Summarizer           | `Falconsai/text_summarization`    |
| Meeting Summary             | `knkarthick/MEETING_SUMMARY`      |
| LLaMA 2 (7B HF)             | `meta-llama/Llama-2-7b-hf`         |

You can easily add more Hugging Face models to this list in the app code.

---

## 🚀 Getting Started

### 1. Clone the Repository
git clone https://github.com/yourusername/webpage-summarizer-app.git
cd webpage-summarizer-app
2. Create and Activate a Virtual Environment
python -m venv venv
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Or manually:
pip install streamlit transformers torch newspaper3k
4. Run the App
🌐 Using the App
Open http://localhost:xxx in your browser.

🧠 Notes
Larger models like LLaMA may need a Hugging Face access token and significant RAM/GPU.

You can swap or add new models in the model_options dictionary in app.py.

📁 Project Structure
webpage-summarizer-app/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file

📜 License
MIT License




