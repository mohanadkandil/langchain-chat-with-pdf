from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type = "pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # loop through pages and extract text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split texts into chunks
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )

        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question in your pdf")
        if user_question:
            docs = base.similarity_search(user_question)
            # answer question based on the llm
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
            




if __name__ == "__main__":
    main()