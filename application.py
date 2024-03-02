import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from loadingtodb import LoadToDB

embedder=HuggingFaceInferenceAPIEmbeddings(
    api_key='Add your API key',
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

os.environ['OPENAI_API_KEY'] = 'Add your API key'


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2== 0:
            st.subheader(message.content)
        else:
            st.write(message.content)

            
def main():

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    #st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")

        if not os.path.exists('Documents'):
            os.makedirs('Documents')

        pdf_docs = st.file_uploader("Upload your PDFs here")
        if pdf_docs is not None:
            with open(os.path.join('Documents', pdf_docs.name), "wb") as f:
                f.write(pdf_docs.getbuffer())
                
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                data=LoadToDB(embedder,'Documents/','Database/',3300,300)
                data.load()

                # get the text chunks
                data.chunk()

                # create vector store
                vectorstore = data.database()

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
            st.success("File Processed Successfully!!!")


if __name__ == '__main__':
    main()