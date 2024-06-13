# DXC LawBot

DXC LawBot is an interactive chatbot application that allows users to ask questions about legal documents. The chatbot uses natural language processing and machine learning techniques to provide relevant answers based on the content of the uploaded PDF documents.


## Features

- User authentication with registration and login
- Upload and process multiple PDF documents
- Chat with a bot to retrieve information from the uploaded documents
- Store and display chat history
- Download chat history

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository:**
   
   git clone https://github.com/yourusername/RAG_ChatBot.git
   cd RAG_ChatBot

2. **Create and activate a virtual environment:**

 
  python -m venv venv
  # On Windows: .\venv\Scripts\activate

3. **Install the dependencies:**
   pip install -r requirements.txt

4. **Set up environment variables:**
  Create a .env file in the root directory and add the following variables:
  OPENAI_API_KEY=your_openai_api_key
