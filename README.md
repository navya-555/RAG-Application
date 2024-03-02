# Chat with Multiple PDFs

This project allows you to chat with multiple PDF documents using Streamlit and various natural language processing libraries.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run application.py
    ```

## Usage

1. Upload your PDF documents in the sidebar.

2. Enter your question in the text input box.

3. Click the "Process" button to generate a response.

## Features

- **Chat with Multiple PDFs**: Engage in a conversation about the contents of uploaded PDF documents.
- **Natural Language Processing**: Utilizes advanced NLP models to understand and respond to user queries.
- **Interactive Interface**: Built with Streamlit, providing a user-friendly interface for interaction.
- **Document Processing**: Automatically processes uploaded PDF documents, extracting text and creating a searchable database.

## Working

1. **Document Processing**: Upon uploading PDF documents, the application automatically extracts text from them using the Langchain Community library's `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.

2. **Vectorization**: The extracted text is chunked and converted into vectors using the Hugging Face `transformers` library, specifically the `HuggingFaceInferenceAPIEmbeddings`.

3. **Conversation Setup**: A conversational chain is established using Langchain Community's `ConversationalRetrievalChain` and `ChatOpenAI` to facilitate user interaction.

4. **User Interaction**: Users can enter questions about the uploaded documents in the text input box. The application then utilizes the conversational chain to generate responses based on the content of the documents.

