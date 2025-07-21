"""HR Resume Analysis RAG Agent - Simplified Flask Application."""
import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from dotenv import load_dotenv
from services.document_service import load_documents, create_vector_store, get_vector_store, delete_all_documents
from services.chat_service import process_chat_question

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/update_vectorstore', methods=['POST'])
def update_vectorstore():
    """Update vector store with new documents."""
    file_path = request.form.get('file_path', '').strip() or os.environ.get("FILE_PATH", "")
    
    if not file_path or not os.path.exists(file_path):
        flash(f"Invalid path: {file_path}", "error")
        return redirect(url_for('index'))
    
    try:
        # Clear existing documents
        if delete_all_documents():
            flash("Existing documents cleared.", "info")
        
        # Load and process new documents
        documents = load_documents(file_path)
        if not documents:
            flash("No documents found.", "warning")
            return redirect(url_for('index'))
        
        vectorstore = create_vector_store(documents)
        if vectorstore:
            flash(f"Successfully updated with {len(documents)} documents.", "success")
        else:
            flash("Error creating vector store.", "error")
            
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        logger.error(f"Update error: {e}")
    
    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        question = request.json.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        vectorstore = get_vector_store()
        if not vectorstore:
            return jsonify({"error": "Vector store not available. Please update first."}), 400
        
        response = process_chat_question(question, vectorstore)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "HR Resume Analysis RAG Agent"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
