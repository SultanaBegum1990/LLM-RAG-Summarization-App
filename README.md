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

```bash
git clone https://github.com/yourusername/webpage-summarizer-app.git
cd webpage-summarizer-app
2. Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv venv
On Windows:
bash
Copy
Edit
venv\Scripts\activate
On macOS/Linux:
bash
Copy
Edit
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install streamlit transformers torch newspaper3k
4. Run the App
bash
Copy
Edit
streamlit run app.py
🌐 Using the App
Open http://localhost:8501 in your browser.

Choose to summarize from a webpage URL or paste text.

Select a summarization model from the dropdown.

Click Summarize and view the result.

🧠 Notes
Larger models like LLaMA may need a Hugging Face access token and significant RAM/GPU.

You can swap or add new models in the model_options dictionary in app.py.

📁 Project Structure
bash
Copy
Edit
webpage-summarizer-app/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file
🛠 Troubleshooting
ModuleNotFoundError: Run pip install for any missing library.

PermissionError on activation (Windows PowerShell): Run Set-ExecutionPolicy RemoteSigned in PowerShell as admin.

📜 License
MIT License

✨ Credits
Built with:

Streamlit

Hugging Face Transformers

Newspaper3k

sql
Copy
Edit

✅ **Tip**: Paste this into your `README.md` file in the root of your repo.  
Then commit and push:

```bash
git add README.md
git commit -m "Add full README with code blocks"
git push
Let me know if you'd like me to help add a screenshot, deploy to Hugging Face Spaces, or include GPU setup tips!
