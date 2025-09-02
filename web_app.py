#!/usr/bin/env python3
"""
SRM AI Doc Assist - Web Application
Enhanced web interface for Dell SRM RAG System
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid

# Import the existing RAG system
from app import DellSRMRAG, RAGConfig

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'srm-ai-doc-assist-secret-key-2024')
app.config['UPLOAD_FOLDER'] = './documents'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize SocketIO for real-time chat
socketio = SocketIO(app, cors_allowed_origins="*")

# Global RAG system instance
rag_system = None
chat_sessions = {}

# Sample data for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "name": "SRM Upgrade Guide.pdf",
        "pages": 42,
        "uploaded": "2 hours ago",
        "type": "pdf",
        "size": "2.3 MB"
    },
    {
        "id": "doc_2", 
        "name": "Technical Specification.pdf",
        "pages": 18,
        "uploaded": "1 day ago",
        "type": "pdf",
        "size": "1.1 MB"
    },
    {
        "id": "doc_3",
        "name": "Project Proposal.pdf",
        "pages": 25,
        "uploaded": "3 days ago",
        "type": "pdf",
        "size": "1.8 MB"
    },
    {
        "id": "doc_4",
        "name": "Research Paper.pdf",
        "pages": 67,
        "uploaded": "1 week ago",
        "type": "pdf",
        "size": "3.2 MB"
    },
    {
        "id": "doc_5",
        "name": "Annual Report.pdf",
        "pages": 89,
        "uploaded": "2 weeks ago",
        "type": "pdf",
        "size": "4.1 MB"
    }
]

SAMPLE_CHATS = [
    {
        "id": "chat_1",
        "title": "SRM Installation Questions",
        "document": "SRM Upgrade Guide.pdf",
        "last_message": "2 hours ago",
        "message_count": 15
    },
    {
        "id": "chat_2",
        "title": "Configuration Help",
        "document": "Technical Specification.pdf", 
        "last_message": "1 day ago",
        "message_count": 8
    },
    {
        "id": "chat_3",
        "title": "Troubleshooting Support",
        "document": "SRM Upgrade Guide.pdf",
        "last_message": "3 days ago",
        "message_count": 12
    }
]

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system
    try:
        # Create configuration
        config = RAGConfig(
            pdf_directory="./documents",
            vector_db_path="./vector_db",
            llm_model="llama3.2:3b",  # Using preferred model
            embedding_model="nomic-embed-text"
        )
        
        # Initialize RAG system
        rag_system = DellSRMRAG()
        rag_system.config = config
        
        if rag_system.initialize():
            app.logger.info("RAG system initialized successfully")
            return True
        else:
            app.logger.error("Failed to initialize RAG system")
            return False
            
    except Exception as e:
        app.logger.error(f"Error initializing RAG system: {e}")
        return False

def get_time_ago(timestamp_str):
    """Convert timestamp string to relative time"""
    try:
        # Parse the timestamp string
        if "hours ago" in timestamp_str:
            hours = int(timestamp_str.split()[0])
            return datetime.now() - timedelta(hours=hours)
        elif "days ago" in timestamp_str:
            days = int(timestamp_str.split()[0])
            return datetime.now() - timedelta(days=days)
        elif "weeks ago" in timestamp_str:
            weeks = int(timestamp_str.split()[0])
            return datetime.now() - timedelta(weeks=weeks)
        else:
            return datetime.now()
    except:
        return datetime.now()

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/documents')
def get_documents():
    """Get list of documents"""
    try:
        # Check if documents directory exists
        docs_dir = Path(app.config['UPLOAD_FOLDER'])
        if docs_dir.exists():
            # Get actual documents
            actual_docs = []
            for pdf_file in docs_dir.glob("*.pdf"):
                # Get file stats
                stat = pdf_file.stat()
                uploaded_time = datetime.fromtimestamp(stat.st_mtime)
                time_diff = datetime.now() - uploaded_time
                
                if time_diff.days > 0:
                    uploaded_str = f"{time_diff.days} days ago"
                elif time_diff.seconds > 3600:
                    hours = time_diff.seconds // 3600
                    uploaded_str = f"{hours} hours ago"
                else:
                    uploaded_str = "Just now"
                
                actual_docs.append({
                    "id": str(uuid.uuid4()),
                    "name": pdf_file.name,
                    "pages": "Unknown",  # Would need PDF processing to get actual page count
                    "uploaded": uploaded_str,
                    "type": "pdf",
                    "size": f"{stat.st_size / (1024*1024):.1f} MB"
                })
            
            return jsonify(actual_docs if actual_docs else SAMPLE_DOCUMENTS)
        else:
            return jsonify(SAMPLE_DOCUMENTS)
            
    except Exception as e:
        app.logger.error(f"Error getting documents: {e}")
        return jsonify(SAMPLE_DOCUMENTS)

@app.route('/api/chats')
def get_chats():
    """Get list of recent chats"""
    try:
        # Return sample chats for now
        # In production, this would come from a database
        return jsonify(SAMPLE_CHATS)
    except Exception as e:
        app.logger.error(f"Error getting chats: {e}")
        return jsonify([])

@app.route('/api/chat/<chat_id>')
def get_chat_history(chat_id):
    """Get chat history for a specific chat"""
    try:
        # Return sample chat history
        # In production, this would come from a database
        sample_history = [
            {
                "id": "msg_1",
                "type": "user",
                "content": "How do I install Dell SRM?",
                "timestamp": "2 hours ago"
            },
            {
                "id": "msg_2", 
                "type": "assistant",
                "content": "To install Dell SRM, you'll need to follow these steps:\n\n1. Ensure your system meets the minimum requirements\n2. Download the installation package\n3. Run the installer with appropriate permissions\n4. Configure the initial settings\n\nWould you like me to provide more specific details about any of these steps?",
                "timestamp": "2 hours ago"
            }
        ]
        return jsonify(sample_history)
    except Exception as e:
        app.logger.error(f"Error getting chat history: {e}")
        return jsonify([])

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a query using the RAG system"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        chat_id = data.get('chat_id', 'new')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Check if RAG system is available
        if not rag_system:
            return jsonify({
                "answer": "I apologize, but the AI system is currently unavailable. Please try again later.",
                "sources": [],
                "data_available": False
            })
        
        # Process query
        result = rag_system.query(question)
        
        # Format response for web interface
        response = {
            "answer": result.get("answer", "No answer available"),
            "sources": result.get("sources", []),
            "data_available": result.get("data_available", False),
            "query_time": result.get("query_time", 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store in chat session if chat_id provided
        if chat_id != 'new':
            if chat_id not in chat_sessions:
                chat_sessions[chat_id] = []
            chat_sessions[chat_id].append({
                "type": "user",
                "content": question,
                "timestamp": response["timestamp"]
            })
            chat_sessions[chat_id].append({
                "type": "assistant", 
                "content": response["answer"],
                "timestamp": response["timestamp"]
            })
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return jsonify({
            "error": "An error occurred while processing your question",
            "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
            "sources": [],
            "data_available": False
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Handle document upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            # Save file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Process document with RAG system if available
            if rag_system:
                try:
                    # This would trigger document processing
                    # For now, just return success
                    pass
                except Exception as e:
                    app.logger.warning(f"Document processing failed: {e}")
            
            return jsonify({
                "success": True,
                "message": f"Document {filename} uploaded successfully",
                "filename": filename
            })
        else:
            return jsonify({"error": "Only PDF files are supported"}), 400
            
    except Exception as e:
        app.logger.error(f"Error uploading document: {e}")
        return jsonify({"error": "Upload failed"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_system": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    })

# SocketIO events for real-time chat
@socketio.on('join')
def on_join(data):
    """Join a chat room"""
    room = data['room']
    join_room(room)
    emit('status', {'msg': f'Joined room: {room}'}, room=room)

@socketio.on('leave')
def on_leave(data):
    """Leave a chat room"""
    room = data['room']
    leave_room(room)
    emit('status', {'msg': f'Left room: {room}'}, room=room)

@socketio.on('message')
def on_message(data):
    """Handle real-time messages"""
    room = data['room']
    emit('message', data, room=room)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    if initialize_rag_system():
        print("✅ RAG system initialized successfully")
    else:
        print("⚠️ RAG system initialization failed - running in demo mode")
    
    # Create documents directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the application
    print("🚀 Starting SRM AI Doc Assist Web Application...")
    print("📱 Open your browser and navigate to: http://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
