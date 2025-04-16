Webpage Summarizer App
A Streamlit-based web application that extracts text from a webpage or user input and summarizes it using Hugging Face Transformers like BART, T5, Pegasus, and LLaMA.

🔍 Features
🌐 Summarize content from any article URL.

📄 Summarize manually pasted text.

🤖 Choose from multiple state-of-the-art Hugging Face summarization models.

⚡ GPU support (if available) for faster inference.

🚀 Demo
Run the app locally and access it at http://localhost:8501

🧠 Supported Models

Model Name	Hugging Face ID
BART	facebook/bart-large-cnn
T5	t5-base
Pegasus	google/pegasus-xsum
LLaMA 2 (7B)	meta-llama/Llama-2-7b-hf
You can expand this list by adding any summarization-capable model from Hugging Face.

⚙️ Setup Instructions
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/webpage-summarizer-app.git
cd webpage-summarizer-app
2. Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install streamlit transformers torch newspaper3k
4. Run the app
bash
Copy
Edit
streamlit run app.py
📦 Project Structure
bash
Copy
Edit
webpage-summarizer-app/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
💡 Usage
Open the app in your browser.

Choose between URL or text input.

Select your preferred summarization model.

Click to generate the summary.

Enjoy the result!

⚠️ Notes
If using large models like LLaMA, ensure you have enough memory/GPU.

Some models may require a Hugging Face token for access (e.g., LLaMA-2).

📖 License
MIT License

