# Conversational_ai_bot

# Conversational AI Search Bot

A powerful local chatbot that combines GPT-2 language generation with semantic document search using FAISS vector embeddings. This project enables intelligent conversations about your documents without requiring external API calls.

## 🚀 Features

- **Local AI Inference**: Uses GPT-2 model for conversation generation (no API keys required)
- **Semantic Document Search**: Leverages SentenceTransformers for intelligent document retrieval
- **FAISS Vector Store**: High-performance vector similarity search for large document collections
- **Interactive UI**: Clean and responsive Streamlit-based chat interface
- **Document Chunking**: Intelligent text segmentation for better context retrieval
- **Session Management**: Maintains conversation history and context
- **Custom Styling**: Enhanced UI with CSS styling for better user experience

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework for the chat interface
- **Hugging Face Transformers**: GPT-2 model for text generation
- **SentenceTransformers**: Semantic text embeddings
- **FAISS**: Facebook AI Similarity Search for vector operations
- **LangChain**: Document processing and chunking framework

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## ⚡ Installation

1. **Clone the repository**
git clone https://github.com/Kalyan-818/conversational_ai_bot.git
cd conversational_ai_bot
2. **Create and activate virtual environment**
python -m venv ai

On Windows:
ai\Scripts\activate
3. **Install dependencies**
pip install -r requirements.txt

text

## 🎯 Usage

1. **Start the application**
streamlit run streamlit_app.py

text

2. **Access the interface**
- Open your web browser
- Navigate to `http://localhost:8501`

3. **Upload documents**
- Add your text documents to the `docs/` directory
- The system will automatically process and index them

4. **Start chatting**
- Type your questions in the chat interface
- The bot will search relevant documents and generate contextual responses

## 📁 Project Structure

conversational_ai_bot/
├── streamlit_app.py # Main Streamlit application
├── main.py # Core application logic
├── requirements.txt # Python dependencies
├── style.css # Custom CSS styling
├── launch.json # VS Code launch configuration
├── docs/ # Document directory (create and add your files)
├── pyvenv.cfg # Virtual environment configuration
└── README.md # Project documentation

text

## 🔧 Key Components

### Document Processing
- **Text Chunking**: Splits documents into manageable segments
- **Embedding Generation**: Creates semantic embeddings using SentenceTransformers
- **Vector Indexing**: Stores embeddings in FAISS for fast retrieval

### Conversation Engine
- **Context Retrieval**: Finds relevant document chunks based on user queries
- **Response Generation**: Uses GPT-2 to generate coherent responses
- **Memory Management**: Maintains conversation context across interactions

## 🎨 Features in Detail

### Semantic Search
The bot uses advanced NLP techniques to understand the meaning behind your questions, not just keyword matching.

### Local Processing
All AI operations happen locally on your machine, ensuring privacy and eliminating API costs.

### Scalable Vector Search
FAISS enables efficient similarity search even with large document collections.

## 🚀 Use Cases

- **Research Assistant**: Query large document collections for specific information
- **Knowledge Base**: Create a conversational interface for company documentation
- **Study Tool**: Interactive learning from textbooks and academic papers
- **Content Analysis**: Analyze and extract insights from document repositories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Gujjula Kalyan Reddy**
- GitHub: [@Kalyan-818](https://github.com/Kalyan-818)
- LinkedIn: [Kalyan Reddy](https://www.linkedin.com/in/kalyan-reddy-0578872a7/)
- Email: gkalyanreddy22@ifheindia.org

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Facebook AI Research for FAISS
- Streamlit team for the amazing web framework
- LangChain for document processing utilities

## 📊 Project Status

✅ **Complete Features:**
- Local GPT-2 integration
- FAISS vector search
- Streamlit UI
- Document chunking
- Session management

🔄 **Future Enhancements:**
- Support for more document formats (PDF, DOCX)
- Advanced conversation memory
- Multi-language support
- Document upload via UI

---

*Built with ❤️ for intelligent document interaction*
