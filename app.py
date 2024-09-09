import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

import os
headers = {
    "authentication": st.secrets["OPENAI_API_KEY"]
}
def main():
    st.title("Chat with EduLink 📚")

    # Load environment variables
    load_dotenv()

    # Upload a PDF file (Optional)
    st.subheader("Upload Your PDF File")
    pdf = st.file_uploader("Drag and drop your PDF file here or click to browse.", type='pdf')

    text = ""
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

    # Accept user queries
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")

    if st.button("Enter"):
        if query:
            try:
                if text:
                    # Process PDF content
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)

                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                    docs = vector_store.similarity_search(query=query, k=3)

                    if docs:
                        # Get answer from the PDF
                        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=query)
                            st.write(response)
                    else:
                        # Get answer from ChatGPT if no relevant content in PDF
                        st.write("Answer not found in the PDF...")
                        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                        response = llm.predict(query)
                        st.write(response)
                else:
                    # If no PDF uploaded, default to ChatGPT response
                    st.write("No PDF uploaded. ChatGPT will answer your question.")
                    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                    response = llm.predict(query)
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
        else:
            st.warning("Please enter a question before pressing 'Enter'.")

if __name__ == '__main__':
    main()
