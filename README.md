# RAG to IAs

An interactive chatbot built for answering questions about the EECS 487 Natural Language Processing course. 
This system combines retrieval-augmented generation (RAG) using course materials (PDFs) with large language models (Gemini or Mistral) and a simple Streamlit interface.

## Features

- 💬 Ask questions about the course syllabus and lecture notes
- 🤖 Supports both **Gemini 2.0 Flash** and **Mistral Large** models
- 🖼️ Retrieves and displays relevant PDF pages as images using ColPaLi
- 🧠 Generates concise, contextual answers with RAG


## Project Structure
```
├── ask_ia.sh            # Script to launch the chatbot
├── mistral_ui.py        # Streamlit UI logic with both models
├── ui.py                # Streamlit UI logic with Gemini
├── rag_pipeline.py      # Backend RAG logic and model handling
├── finetune.ipynb       # Notebook to finetune embeddings
├── eecs-487-docs/       # Folder for PDF course materials (input)
├── images/              # Images for similarity maps
└── data/                # Folder for image conversion (optional)

```

## Setup

#### 1. Clone the repository
```bash
git clone https://github.com/OGaCu/rags-to-ias.git
cd rags-to-ias
```

#### 2. Install dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install -r requirements.txt
```

You may also need to install Poppler for pdf2image:
- On macOS: `brew install poppler`
- On Ubuntu: `sudo apt-get install poppler-utils`

## API Keys
Feel free to use your own API keys if you have them. 
This allows you to customize the experience and use your own quotas.
Edit the following variables in `rag_pipeline.py` to set your API keys:
```python
GOOGLE_API_KEY = "<your-gemini-api-key>"
MISTRAL_API_KEY = "<your-mistral-api-key>"
```

## Run the Chatbot
Ensure that your PDF course materials are placed in the `eecs-487-docs/` directory.

Then launch the chatbot with:
```bash
./ask_ia.sh
```

This command starts the Streamlit app and opens it in your browser.

## Example Queries
Try questions like:
- "What is the grading policy?"
- "Explain the attention mechanism from lecture 13."
- "Which models were covered in the transformer lecture?"

## Notes
- If a query is matched with visual content, an image will be shown from the relevant PDF page.
- Image display requires the top document score to exceed a threshold.
- You can toggle between Gemini and Mistral models in the app interface from the sidebar.

## Troubleshooting
- **No PDFs found**: Make sure they exist under `eecs-487-docs/` with the correct names.
- **Images not showing**: Ensure Poppler is installed and images are being extracted correctly.
- **App not loading**: Ensure all Python dependencies and API keys are properly set.

## License
MIT License – feel free to use, modify, and share.

## Acknowledgements
- LangChain
- FAISS by Facebook AI
- Google Gemini
- Mistral
- ColPaLi
