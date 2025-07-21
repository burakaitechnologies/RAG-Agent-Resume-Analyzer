# HR Resume Analysis RAG Agent

A clean, Pythonic Flask-based Retrieval-Augmented Generation (RAG) application for HR professionals to analyze resumes, job descriptions, and conduct intelligent hiring assessments.

## Features

- **Document Processing**: Supports .txt, .pdf, and .docx files
- **Vector Store Management**: Easy document indexing with Pinecone
- **Intelligent Chat**: HR-focused analysis and recommendations  
- **Source Attribution**: See which documents informed each response

## Project Structure

```
├── app.py                 # Main Flask application
├── document_service.py    # Document loading and vector store operations
├── chat_service.py        # RAG chain and chat logic
├── requirements.txt       # Dependencies
└── templates/
    └── index.html         # Web interface
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RAGAgent-ResumeAnalyzer.git
cd RAGAgent-ResumeAnalyzer
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux  
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file with your actual API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=embeddings-index
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=hr-resume-analyzer
FILE_PATH=./sample_documents
```

### 5. Prepare Your Documents

Create a directory with your HR documents:
- Resumes (.pdf, .docx, .txt)
- Job descriptions
- HR policies
- Interview guides
- Salary benchmarks

### 6. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Update Vector Store
1. Navigate to the "Update Vector Store" section
2. Enter the path to your documents (or leave empty to use the default path from .env)
3. Click "Update Vector Store" to process and index your documents

### Chat with the HR Assistant
Ask questions like:
- "Evaluate this candidate's qualifications for the software engineer position"
- "What are the key skills mentioned in these resumes?"
- "Compare the top 3 candidates for this role"
- "What interview questions should I ask for this position?"
- "What is the salary range for this role based on the documents?"

## Supported File Types

- **Text files** (.txt): Plain text resumes, job descriptions
- **PDF files** (.pdf): Resume PDFs, formatted documents
- **Word documents** (.docx): Microsoft Word resumes and documents

## Key Features for HR Professionals

### Resume Analysis
- Skills extraction and evaluation
- Experience relevance assessment
- Education background review
- Career progression analysis

### Job Matching
- Candidate-role alignment scoring
- Skills gap identification
- Qualification requirements mapping
- Cultural fit indicators

### Interview Support
- Targeted question generation
- Competency-based interview guides
- Technical skill assessment suggestions

### Compensation Analysis
- Salary benchmarking queries
- Market rate comparisons
- Benefits package evaluation

## Technical Architecture

- **Flask**: Web framework for the user interface
- **LangChain**: Document processing and RAG pipeline
- **OpenAI**: Large language model for intelligent responses
- **Pinecone**: Vector database for document embeddings
- **Bootstrap**: Responsive UI framework

## Security Notes

- Keep your API keys secure and never commit them to version control
- Use environment variables for all sensitive configuration
- Consider implementing authentication for production use
- Regularly rotate API keys and monitor usage

## API Keys Setup

### Required APIs:

1. **OpenAI API**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone API**: Sign up at [Pinecone](https://www.pinecone.io/) and get your API key
3. **LangChain API** (Optional): For tracing, get key from [LangSmith](https://smith.langchain.com/)

## Troubleshooting

### Common Issues

1. **Vector store not available**: Make sure you've updated the vector store with documents first
2. **File loading errors**: Check that file paths are correct and files are in supported formats
3. **API errors**: Verify your API keys are valid and have sufficient credits/quota

### Logs

The application logs important events and errors. Check the console output for debugging information.

## Contributing

Feel free to enhance the application with additional features like:
- User authentication
- Document upload via web interface
- Advanced filtering and search
- Integration with ATS systems
- Batch processing capabilities
