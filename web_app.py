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
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, Response
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

# Chat history management

CHAT_STORAGE_FILE = './data/chat_history.json'
CHAT_SESSIONS_FILE = './data/chat_sessions.json'

def ensure_data_directory():
    """Ensure data directory exists"""
    os.makedirs('./data', exist_ok=True)

def load_chat_data():
    """Load chat data from persistent storage"""
    ensure_data_directory()

    chat_data = {
        'chats': [],
        'chat_history': {},
        'sessions': {}
    }

    # Load chats
    if os.path.exists(CHAT_STORAGE_FILE):
        try:
            with open(CHAT_STORAGE_FILE, 'r') as f:
                chat_data['chats'] = json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading chat data: {e}")
            chat_data['chats'] = []

    # Load chat history
    history_file = './data/chat_history_data.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                chat_data['chat_history'] = json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading chat history: {e}")
            chat_data['chat_history'] = {}

    # Load sessions
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, 'r') as f:
                chat_data['sessions'] = json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading sessions: {e}")
            chat_data['sessions'] = {}

    return chat_data

def save_chat_data(chat_data):
    """Save chat data to persistent storage"""
    ensure_data_directory()

    try:
        # Save chats
        with open(CHAT_STORAGE_FILE, 'w') as f:
            json.dump(chat_data['chats'], f, indent=2)

        # Save chat history
        with open('./data/chat_history_data.json', 'w') as f:
            json.dump(chat_data['chat_history'], f, indent=2)

        # Save sessions
        with open(CHAT_SESSIONS_FILE, 'w') as f:
            json.dump(chat_data['sessions'], f, indent=2)

    except Exception as e:
        app.logger.error(f"Error saving chat data: {e}")

# Global chat data
chat_data = load_chat_data()

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

SAMPLE_CHATS = []

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

def update_chat_metadata(chat_id, last_message):
    """Update chat metadata (message count, last message, etc.)"""
    try:
        chats = chat_data['chats']
        chat = next((c for c in chats if c['id'] == chat_id), None)

        if chat:
            chat['last_message'] = last_message[:50] + '...' if len(last_message) > 50 else last_message
            chat['message_count'] = len(chat_data['chat_history'].get(chat_id, []))
            chat['updated_at'] = datetime.now().timestamp() * 1000  # Store as milliseconds
        else:
            # Create new chat entry if it doesn't exist
            new_chat = {
                'id': chat_id,
                'title': f'Chat {datetime.now().strftime("%H:%M")}',
                'document': 'General',
                'last_message': last_message[:50] + '...' if len(last_message) > 50 else last_message,
                'message_count': len(chat_data['chat_history'].get(chat_id, [])),
                'created_at': datetime.now().timestamp() * 1000,
                'updated_at': datetime.now().timestamp() * 1000
            }
            chats.append(new_chat)

        # Sort chats by updated_at (most recent first)
        chats.sort(key=lambda x: x.get('updated_at', 0), reverse=True)

    except Exception as e:
        app.logger.error(f"Error updating chat metadata: {e}")

@app.route('/api/chat', methods=['POST'])
def create_chat():
    """Create a new chat session."""
    try:
        data = request.json
        chat_id = data.get('id')
        title = data.get('title')

        if not chat_id or not title:
            return jsonify({"error": "Missing chat_id or title"}), 400

        new_chat = {
            "id": chat_id,
            "title": title,
            "document": "General",
            "last_message": "Chat created",
            "message_count": 0,
            "created_at": datetime.now().isoformat()
        }
        chat_data['chats'].insert(0, new_chat)
        chat_data['chat_history'][chat_id] = []
        save_chat_data(chat_data)
        
        return jsonify(new_chat), 201
    except Exception as e:
        app.logger.error(f"Error creating chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat session."""
    try:
        chat_data['chats'] = [chat for chat in chat_data['chats'] if chat['id'] != chat_id]
        if chat_id in chat_data['chat_history']:
            del chat_data['chat_history'][chat_id]
        save_chat_data(chat_data)
        return jsonify({"message": "Chat deleted"}), 200
    except Exception as e:
        app.logger.error(f"Error deleting chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/documents/<path:filename>')
def serve_document(filename):
    """Serve documents from the documents directory"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
        # Return stored chats, or empty list if none exist
        chats = chat_data['chats']
        return jsonify(chats)
    except Exception as e:
        app.logger.error(f"Error getting chats: {e}")
        return jsonify([])

@app.route('/api/chat/<chat_id>')
def get_chat_history(chat_id):
    """Get chat history for a specific chat"""
    try:
        # Get stored chat history, or return empty array for new chats
        chat_history_data = chat_data['chat_history']

        if chat_id in chat_history_data and chat_history_data[chat_id]:
            return jsonify(chat_history_data[chat_id])

        # Return empty array for new chats
        return jsonify([])
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
        
        # Store in chat history if chat_id provided
        if chat_id != 'new':
            if chat_id not in chat_data['chat_history']:
                chat_data['chat_history'][chat_id] = []

            # Add user message
            chat_data['chat_history'][chat_id].append({
                "id": f"msg_{len(chat_data['chat_history'][chat_id]) + 1}",
                "type": "user",
                "content": question,
                "timestamp": response["timestamp"]
            })

            # Add assistant message
            chat_data['chat_history'][chat_id].append({
                "id": f"msg_{len(chat_data['chat_history'][chat_id]) + 1}",
                "type": "assistant",
                "content": response["answer"],
                "timestamp": response["timestamp"],
                "sources": result.get("sources", []),
                "responseTime": result.get("query_time", 0) * 1000,  # Convert to milliseconds
                "modelUsed": "llama3.2:3b"
            })

            # Update chat metadata
            update_chat_metadata(chat_id, question)

            # Save to persistent storage
            save_chat_data(chat_data)
        
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

@app.route('/api/chat/clear', methods=['POST'])
def clear_all_chats():
    """Clear all chats and chat history"""
    try:
        app.logger.info("Starting to clear all chats...")
        
        # Clear all data
        chat_data['chats'] = []
        chat_data['chat_history'] = {}
        chat_data['sessions'] = {}
        
        app.logger.info("Chat data cleared in memory")
        
        # Save empty data
        save_chat_data(chat_data)
        
        app.logger.info("Chat data saved to files successfully")
        
        response_data = {
            "success": True,
            "message": "All chats cleared successfully"
        }
        
        app.logger.info("Sending success response")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error clearing chats: {e}")
        return jsonify({"error": "Failed to clear chats"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_system": rag_system is not None,
        "chat_count": len(chat_data['chats']),
        "total_messages": sum(len(messages) for messages in chat_data['chat_history'].values()),
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

@socketio.on('stream_query')
def handle_stream_query(data):
    """Handle a streaming query from the client"""
    question = data.get('question', '').strip()
    chat_id = data.get('chat_id')
    sid = request.sid

    if not question or not rag_system:
        emit('stream_error', {'error': 'Invalid request or RAG system not ready'}, room=sid)
        return

    try:
        full_response = ""
        for response_part in rag_system.stream_query(question):
            if response_part['type'] == 'token':
                token = response_part['content']
                full_response += token
                emit('stream_token', {'token': token}, room=sid)
            
            elif response_part['type'] == 'end':
                # Save the full response to chat history
                if chat_id:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add user message
                    if chat_id not in chat_data['chat_history']:
                        chat_data['chat_history'][chat_id] = []
                    
                    chat_data['chat_history'][chat_id].append({
                        "id": f"msg_{len(chat_data['chat_history'][chat_id]) + 1}",
                        "type": "user",
                        "content": question,
                        "timestamp": timestamp
                    })
                    
                    # Add full assistant message
                    chat_data['chat_history'][chat_id].append({
                        "id": f"msg_{len(chat_data['chat_history'][chat_id]) + 1}",
                        "type": "assistant",
                        "content": full_response,
                        "timestamp": timestamp,
                        "sources": response_part['sources'],
                        "modelUsed": rag_system.config.llm_model,
                        "performance": response_part.get('performance', {})
                    })
                    
                    update_chat_metadata(chat_id, question)
                    save_chat_data(chat_data)
                
                # Emit performance metrics along with sources
                performance_data = response_part.get('performance', {})
                emit('stream_end', {
                    'sources': response_part['sources'],
                    'performance': performance_data
                }, room=sid)

            elif response_part['type'] == 'error':
                emit('stream_error', {'error': response_part['content']}, room=sid)

    except Exception as e:
        app.logger.error(f"Error during stream_query: {e}")
        emit('stream_error', {'error': 'An internal error occurred during streaming.'}, room=sid)


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
