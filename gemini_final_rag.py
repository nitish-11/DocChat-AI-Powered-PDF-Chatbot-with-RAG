import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64
import fitz 


# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for chat history and PDF processing
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# Function to display the PDF of a given file
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=4000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = HuggingFaceEmbeddings(
    #              model_name="BAAI/bge-large-en-v1.5",
    #              model_kwargs={"device": "cpu"}  # Ensures it's not downloading locally
    # )
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain():
    # prompt_template = """
    # Answer the question as detailed as possible using the provided context and previous conversation history.
    # If the answer is not in the provided context, just say, "answer is not available in the context." Do not provide a wrong answer.
    # Also, if question asked trying to compare two different incidents, try to extract out the data and answer the question.
    
    # Conversation History:
    # {history}
    
    # Context:
    # {context}
    
    # Question:
    # {question}
    
    # Answer:
    # """
    
    prompt_template = """You are an intelligent assistant with access to detailed transcripts and conversation history. Your goal is to provide accurate, well-researched answers based on the given context. 
                          Follow these instructions carefully:

            1. **Use the provided context** and **previous conversation history** to answer the question as thoroughly as possible.
            2. **If the answer is not in the provided context, say,** "The answer is not available in the context." **Do not generate false information.**
            3. **If the question requires comparing two different events, characters, or time periods,** analyze the relevant parts carefully before answering.
            4. **For long-term analysis questions (e.g., character development across seasons),** search all relevant text to provide a well-supported response.
            5. **If the question refers to a specific season or time period (e.g., "What was Ross doing in Season 4?"),** traverse all episodes of that season to extract the necessary information.
            
            Remember, if the answer is not in the provided context, just say, "answer is not available in the context." 
            Do not provide a wrong answer.
            ---

            ### **Conversation History:**
            {history}

            ### **Context:**
            {context}

            ### **Question:**
            {question}

            ### **Answer:**
            """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    chain = prompt | model
    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please upload and process a PDF first.")
        return
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # query_embedding = embeddings.embed_query(user_question)  # Convert question into vector
    retriever = new_db.as_retriever()
    # docs = new_db.similarity_search(query_embedding)
    docs = retriever.invoke(user_question)
    chain = get_conversational_chain()
    
    # Include chat history in the prompt
    chat_history_text = "\n".join(st.session_state.chat_history)
    # response = chain({"history": chat_history_text, "input_documents": docs, "question": user_question}, return_only_outputs=True)
    response = chain.invoke({"history": chat_history_text, "context": docs, "question": user_question})
    
    reply = response.content
    
    # Update chat history
    st.session_state.chat_history.append(f"User: {user_question}")
    st.session_state.chat_history.append(f"Model: {reply}")
    
    st.write("Answer:", reply)

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Chat with PDF using Gemini 2.0 Pro💁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    with st.sidebar:
        st.title("Menu:")
        st.markdown("### 📚 Your Personal Document Assistant")
        st.markdown("---")
        menu = ["🏠 Home", "📝 Chat with PDF", "🔍 Backend Logic", "📧 Contact"]
        choice = st.selectbox("Navigate", menu)

    # Home Page
    if choice == "🏠 Home":
        st.title("📝 DocChat: AI-Powered PDF Chatbot with RAG")
        st.markdown("""
        #### 🌟 Welcome Upload PDFs, get summaries, and chat with your documents AI-powered and hassle-free!

        ---
        ### ⚙️ Tech Stack:
        - 📝 **Google Gemini 2.0 Pro** – Delivers intelligent and accurate responses  
        - 📚 **FAISS Vector Search** – Efficient and fast document retrieval  
        - 🔗 **LangChain** – Seamless integration of AI-powered conversations  
        - 🎨 **Streamlit UI** – Smooth and user-friendly interface  

        ---
        ### ✨ Features:
        - 📂 **Upload Documents** – Easily upload PDFs and preview them on-screen  
        - 📑 **Summarize** – Get concise, AI-generated summaries of your content  
        - 💬 **Chat** – Interact with your documents like never before!  

        🔥 **Try it now and experience AI-driven document understanding!** 🔥
        """)

    # Chatbot Page
    elif choice == "📝 Chat with PDF":
        st.title("📝 Chatbot Interface For PDF")
        st.markdown("---")

        # Create two columns
        col1, col2 = st.columns(2)

        # Column 1: File Uploader and Preview
        with col1:
            st.header("📂 Upload Document")
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
            if uploaded_file is not None:
                st.success("📄 File Uploaded Successfully!")
                st.markdown(f"**Filename:** {uploaded_file.name}")
                file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert bytes to MB
                st.markdown(f"**File Size:** {file_size_mb:.2f} MB")
                
                # Display PDF preview
                st.markdown("### 📖 PDF Preview")
                displayPDF(uploaded_file)
                
                # Save the uploaded file to a temporary location
                temp_pdf_path = "temp.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Store the temp_pdf_path in session_state
                st.session_state['temp_pdf_path'] = temp_pdf_path

                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text([uploaded_file])
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.pdf_processed = True
                        st.success("Done")

        # Column 2: Chat Interface
        with col2:
            st.header("📒Chat with the PDF")
            if st.session_state.pdf_processed:
                user_question = st.text_input("Ask a Question from the PDF Files")
                if user_question:
                    user_input(user_question)

                st.subheader("Chat History")
                for message in st.session_state.chat_history:
                    st.text(message)
            else:
                st.warning("Please upload and process a PDF first.")

    elif choice == '🔍 Backend Logic':
        st.title('RAG Chatbot Flowchart')
        st.markdown('---')

        # Display Flowchart
        st.image("photos/systemicview_final.png", caption="RAG Chatbot Flowchart")

        st.title('Embedding Vectors in FAISS')
        st.write("From the 276-page PDF, only 26 embedding vectors were generated due to a chunk size of 20k. This highlights Gemini 2.0's ability to handle long-context inputs, making PDF interactions more flexible and dynamic.")

        # Display Embedding Vector Information
        st.image("photos/number_vectors.png", caption="Embedding Vectors Overview")
        st.image("photos/single_vector.png", caption="Single Embedded Vector")
        st.image("photos/memory_loc_vectors.png", caption="Memory Locations of Embedded Vectors")




    # Contact Page
    elif choice == "📧 Contact":
        st.title("📬 Contact Us")
        st.markdown("""
        We'd love to hear from you! Whether you have a question, feedback, or want to contribute, feel free to reach out.

        - **Email:** [Click to Contact](mailto:nitishgupta3476@gmail.com) ✉️
        - **GitHub:** [Contribute on GitHub](https://github.com/nitish-11/DocChat-AI-Powered-PDF-Chatbot-with-RAG) 🛠️

        If you'd like to request a feature or report a bug, please open a pull request on our GitHub repository. Your contributions are highly appreciated! 🙌
        """)


if __name__ == "__main__":
    main()

