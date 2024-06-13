import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pymongo
import io
from datetime import datetime
import bcrypt
import base64
import os


def setup_environment_and_db():
    load_dotenv()
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["dxc_chat_bot2"]
    col1 = mydb["chunks"]
    col2 = mydb["conversation"]
    col3 = mydb["col_documents_pdf"]
    col4 = mydb["users"]  # New collection for user authentication
    return col1, col2, col3, col4

def register_user(users_collection, name, email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_data = {"name": name, "email": email, "password": hashed_password}
    users_collection.insert_one(user_data)

def authenticate_user(users_collection, email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return user['name']
    else:
        return None

def get_pdf_text(pdf_bytes_list):
    text = ""
    for pdf_bytes in pdf_bytes_list:
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

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


def image_to_base64(image_path):             #reads the image files and convert them to base64 format
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def handle_userinput(user_question, chat_history_collection, user_name):
    response = st.session_state.conversation({'question': user_question})
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    response_dict = {
        "user_name": user_name,
        "question": user_question,
        "response": {
            "chat_history": [msg.content for msg in response['chat_history']],
            "timestamp": current_time
        }
    }

    chat_history_collection.insert_one(response_dict)

    # Ensure the images are correctly referenced using relative paths
    user_icon_path = os.path.join(os.path.dirname(__file__), "user_icon.jpg")
    bot_icon_path = os.path.join(os.path.dirname(__file__), "bot_icon.jpg")

    user_icon_base64 = image_to_base64(user_icon_path)
    bot_icon_base64 = image_to_base64(bot_icon_path)

    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">'
                f'<div style="background-color: #6aa84f; color: #ffffff; padding: 10px; border-radius: 10px; margin-right: 10px;">{message.content}</div>'
                f'<img src="data:image/jpeg;base64,{user_icon_base64}" alt="User Icon" width="50"></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                f'<img src="data:image/jpeg;base64,{bot_icon_base64}" alt="Bot Icon" width="100" style="margin-right: 10px;">'
                f'<div style="background-color: #5F249F; color: #ffffff; padding: 10px; border-radius: 10px;">{message.content}</div></div>', 
                unsafe_allow_html=True
            )

def display_chat_history(chat_history_collection, user_name):
    st.subheader("Chat History")
    for entry in chat_history_collection.find({"user_name": user_name}):
        st.write("---")
        if "timestamp" in entry["response"]:
            st.write("Timestamp:", entry["response"]["timestamp"])
        else:
            st.write("Timestamp: N/A")
        st.write("User:", entry["question"])
        st.write("Dxc LawBot:", entry["response"]["chat_history"][-1])

def download_chat_history(chat_history_collection, user_name):
    chat_history_text = ""
    for entry in chat_history_collection.find({"user_name": user_name}):
        chat_history_text += "User: " + entry["question"] + "\n"
        chat_history_text += "Dxc LawBot: " + entry["response"]["chat_history"][-1] + "\n"
        if "timestamp" in entry["response"]:
            chat_history_text += "Timestamp: " + entry["response"]["timestamp"] + "\n\n"
        else:
            chat_history_text += "Timestamp: N/A\n\n"
    
    return chat_history_text

def main():
    col_chunks, col_conversation, col_documents_pdf, col_users = setup_environment_and_db()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":keyboard:")
    custom_css = '''
    <style>
        body {
            background-color: #ECE3ED;
        }
        .stApp {
            background-color: #ECE3ED;
            padding: 20px;
        }
        .stImage {
            position: absolute;
            top: 20px;
            left: 20px;
        }
        .container {
            max-width: 400px;
            margin: auto;
            text-align: center;
            padding-top: 100px;
        }
        .stButton button {
            background-color: #5F249F;
            color: #FFFFFF;
            width: 100%;
        }
        .stTextInput input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
        }
        .sidebar .stButton button {
            background-color: #6aa84f;
            color: #FFFFFF;
            width: 100%;
            margin-bottom: 10px;
        }
    </style>
    '''
        
    st.markdown(custom_css, unsafe_allow_html=True)
    
    if "user_name" not in st.session_state or not st.session_state.user_name:
        if "register_mode" not in st.session_state:
            st.session_state.register_mode = False

        if not st.session_state.register_mode:
            login_container = st.container()
            with login_container:
                #st.image("logo_dxc.jpg", width=250)
                st.write('<div class="container login">', unsafe_allow_html=True)
                st.image('logo_dxc.jpg')
                st.title("Welcome to DXC LawBot!")
                st.subheader("Login to continue")

                email = st.text_input("EMAIL", key="login_email")
                password = st.text_input("PASSWORD", type="password", key="login_password")

                login_button = st.button("Login")

                if login_button:
                    user_name = authenticate_user(col_users, email, password)
                    if user_name:
                        st.session_state.user_name = user_name
                        st.success(f"Welcome back, {user_name}!")
                    else:
                        st.error("Invalid email or password.")

                st.markdown("Don't have an account? [Register Here](#register)")
                if st.button("Register Here"):
                    st.session_state.register_mode = True
                st.write('</div>', unsafe_allow_html=True)
        else:
            register_container = st.container()
            with register_container:
                #st.image("logo_dxc.jpg", width=100)
                st.write('<div class="container register">', unsafe_allow_html=True)
                st.image('logo_dxc.jpg')
                st.title("Register")
                #st.subheader("to continue")

                full_name = st.text_input("FULL NAME", key="register_full_name")
                email = st.text_input("EMAIL", key="register_email")
                password = st.text_input("PASSWORD", type="password", key="register_password")
                confirm_password = st.text_input("CONFIRM PASSWORD", type="password", key="register_confirm_password")

                register_button = st.button("Register")

                if register_button:
                    if password == confirm_password:
                        register_user(col_users, full_name, email, password)
                        st.success("Registration successful. Please log in.")
                        st.session_state.register_mode = False
                    else:
                        st.error("Passwords do not match.")
                st.write('</div>', unsafe_allow_html=True)


    else:
        # If user is logged in, show sidebar menu and main content
        user_name = st.session_state.user_name
     

        # Sidebar menu
        #st.sidebar.image("logo_dxc.jpg", width=150)
        st.sidebar.image('logo_dxc.jpg')
        st.sidebar.title("Menu")
        sidebar_option = st.sidebar.radio("Select an option", ("New Chat", "Chat History"))

        if sidebar_option == "New Chat":
            st.header(f"Welcome, {user_name}! Ask me anything about your documents.")
            user_question = st.text_input("Your question:", key="user_question")
            if user_question:
                handle_userinput(user_question, col_conversation, user_name)

            st.sidebar.subheader("Upload new documents")
            pdf_docs = st.sidebar.file_uploader("Upload new PDFs here:", accept_multiple_files=True, key="pdf_uploader")

            st.sidebar.subheader("Or select existing documents")
            existing_files = [doc["filename"] for doc in col_documents_pdf.find({}, {"filename": 1})]
            selected_files = st.sidebar.multiselect("Select documents to ask questions about:", existing_files, key="select_files")
            pdf_bytes = [col_documents_pdf.find_one({"filename": filename})["pdf"] for filename in selected_files] if existing_files else None

            if st.sidebar.button("Process", key="process_button"):
                with st.spinner("Processing"):
                    pdf_bytes_list = []  # Initialize pdf_bytes_list as a list
                    if pdf_docs:
                        # Store uploaded pdfs in mongodb
                        for pdf in pdf_docs:
                            filename = pdf.name
                            # Check if file already exists in the db
                            if filename not in existing_files:
                                pdf_content = pdf.read()
                                pdf_bytes_list.append(pdf_content)
                                col_documents_pdf.insert_one({"filename": filename, "pdf": pdf_content})
                                existing_files.append(filename)

                    if pdf_bytes:
                        for filename in selected_files:
                            pdf_content = col_documents_pdf.find_one({"filename": filename})["pdf"]
                            pdf_bytes_list.append(pdf_content)

                    if pdf_bytes_list:
                        raw_text = get_pdf_text(pdf_bytes_list)
                        text_chunks = get_text_chunks(raw_text)

                        for chunk in text_chunks:
                            col_chunks.insert_one({"chunk": chunk})      # Store text chunks in the "chunks" collection

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)

                        st.sidebar.success("Processing completed.")

        elif sidebar_option == "Chat History":
            display_chat_history(col_conversation, user_name)

            if st.sidebar.button("Download Chat History", key="download_button"):
                chat_history_text = download_chat_history(col_conversation, user_name)
                if chat_history_text:
                    st.sidebar.download_button(label="Download", data=chat_history_text, file_name="chat_history.txt", mime="text/plain")

        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.user_name = None

if __name__ == '__main__':
    main()

#all working good , even the icons( but trying to find a simmpler way to solve the icon issue)
#fixed logo issue
# try maybe to reduce whitespace on the top of the page
#Last Modified : 09/06/2024
#JKDFJKG
