# SRM AI Doc Assist - Web Application

A modern, responsive web interface for the Dell SRM RAG (Retrieval-Augmented Generation) system, designed to provide an intuitive way to interact with your Dell SRM documentation using AI.

## 🎨 UI Design Features

The web application implements the exact design you specified with these enhanced features:

### **Left Sidebar (Dark Theme)**
- **Header**: Purple brain icon with "SRM AI Doc Assist" branding
- **New Chat Button**: Prominent gradient button to start fresh conversations
- **Recent Chats**: Collapsible section showing chat history with document context
- **Documents Section**: Lists uploaded PDFs with metadata (pages, upload time, size)
- **Upload PDF**: Easy document upload functionality

### **Right Chat Area (Light Theme)**
- **Chat Header**: Document chat title with settings gear icon
- **AI Introduction**: Welcome message from the AI assistant
- **Message Interface**: Clean chat bubbles with user/assistant distinction
- **Input Field**: Auto-resizing textarea with character count and send button
- **Disclaimer**: AI accuracy notice for important information

### **Enhanced Features**
- **Real-time Chat**: Socket.IO integration for live updates
- **Document Context**: Click documents to start focused conversations
- **Source Attribution**: Shows relevant document sources with similarity scores
- **Settings Modal**: Configure AI model and response style preferences
- **Upload Progress**: Visual feedback during document processing
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ with virtual environment
- Ollama running locally with required models
- PDF documents in the `documents/` folder

### Installation

1. **Activate your virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install web dependencies**:
   ```bash
   pip install flask flask-socketio python-socketio python-engineio
   ```

3. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

4. **Start the web application**:
   ```bash
   python start_web_app.py
   ```

5. **Open your browser** and navigate to: `http://localhost:5000`

## 📱 Usage Guide

### **Starting a New Chat**
- Click the "New Chat" button in the sidebar
- The AI will greet you and ask how it can help

### **Uploading Documents**
- Click "Upload PDF" in the sidebar
- Select your Dell SRM PDF file
- Watch the upload progress
- Documents are automatically processed and indexed

### **Chatting with Documents**
- **General Questions**: Ask about Dell SRM concepts, installation, configuration
- **Document-Specific**: Click on a document to focus the conversation
- **Technical Support**: Get help with troubleshooting, procedures, and best practices

### **Managing Conversations**
- **Recent Chats**: Access previous conversations from the sidebar
- **Document Context**: See which documents were referenced in responses
- **Source Information**: View similarity scores and document excerpts

### **Customizing Settings**
- Click the gear icon in the chat header
- **AI Model**: Choose between different Llama models
- **Response Style**: Select concise, detailed, or technical responses

## 🏗️ Architecture

### **Frontend**
- **HTML5**: Semantic markup with accessibility features
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Vanilla JavaScript**: No heavy frameworks, optimized performance
- **Font Awesome**: Professional icon library
- **Socket.IO Client**: Real-time communication

### **Backend**
- **Flask**: Lightweight Python web framework
- **Flask-SocketIO**: WebSocket support for real-time features
- **Integration**: Seamlessly connects to existing RAG system
- **API Endpoints**: RESTful API for document and chat management

### **AI Integration**
- **RAG System**: Leverages existing Dell SRM RAG infrastructure
- **Ollama Models**: Local LLM processing with llama3.2:3b default
- **Vector Search**: Semantic document retrieval with similarity scoring
- **Context Preservation**: Maintains conversation history and document context

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional: Custom secret key
export SECRET_KEY="your-secret-key-here"

# Optional: Custom port
export FLASK_PORT=5000
```

### **Model Configuration**
The web app uses your preferred `llama3.2:3b` model by default. You can change this in the settings modal or by modifying the configuration in `web_app.py`.

### **Document Processing**
- **Supported Format**: PDF files only
- **File Size Limit**: 50MB maximum
- **Storage Location**: `./documents/` folder
- **Auto-indexing**: Documents are processed when uploaded

## 📊 Features Breakdown

### **Core Functionality**
- ✅ **Document Upload**: Drag & drop or click to upload PDFs
- ✅ **AI Chat**: Natural language queries about your documents
- ✅ **Real-time Updates**: Live chat with typing indicators
- ✅ **Source Attribution**: See which documents support each answer
- ✅ **Chat History**: Persistent conversation management

### **User Experience**
- ✅ **Responsive Design**: Works on all device sizes
- ✅ **Dark/Light Theme**: Professional color scheme
- ✅ **Smooth Animations**: Fade-in effects and transitions
- ✅ **Keyboard Shortcuts**: Enter to send, Shift+Enter for new lines
- ✅ **Auto-scroll**: Chat automatically scrolls to new messages

### **Advanced Features**
- ✅ **Document Context**: Focus conversations on specific documents
- ✅ **Similarity Scoring**: Relevance metrics for source documents
- ✅ **Settings Management**: Customize AI behavior and preferences
- ✅ **Error Handling**: Graceful fallbacks and user-friendly messages
- ✅ **Performance**: Optimized for fast response times

## 🐛 Troubleshooting

### **Common Issues**

1. **Web app won't start**:
   - Check if Flask is installed: `pip install flask`
   - Ensure you're in the correct directory
   - Verify Python version (3.10+)

2. **AI responses not working**:
   - Check if Ollama is running: `ollama serve`
   - Verify models are downloaded: `ollama list`
   - Check the console for error messages

3. **Documents not uploading**:
   - Ensure file is PDF format
   - Check file size (max 50MB)
   - Verify `documents/` folder exists

4. **Chat not responding**:
   - Check browser console for JavaScript errors
   - Verify internet connection for CDN resources
   - Try refreshing the page

### **Debug Mode**
The web app runs in debug mode by default. Check the terminal for detailed error messages and Flask logs.

## 🔒 Security Considerations

- **Local Deployment**: Designed for internal/development use
- **File Validation**: PDF format and size restrictions
- **Input Sanitization**: XSS protection for user inputs
- **Session Management**: Secure session handling
- **CORS Configuration**: Configured for local development

## 🚀 Deployment

### **Development**
```bash
python web_app.py
```

### **Production**
For production deployment, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up reverse proxy (Nginx, Apache)
- Configuring SSL certificates
- Implementing user authentication
- Setting up monitoring and logging

## 📈 Performance

- **Fast Loading**: Optimized CSS and JavaScript
- **Efficient Rendering**: Minimal DOM manipulation
- **Real-time Updates**: WebSocket-based communication
- **Responsive UI**: Smooth animations and transitions
- **Memory Efficient**: Clean event handling and cleanup

## 🤝 Contributing

The web application is designed to be easily extensible:

- **New Features**: Add to the Flask routes and JavaScript functions
- **UI Improvements**: Modify the HTML template and CSS
- **AI Integration**: Extend the RAG system integration
- **Testing**: Add unit tests for new functionality

## 📄 License

This web application is part of the Dell SRM RAG System and follows the same licensing terms.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console logs for error messages
3. Ensure all dependencies are properly installed
4. Verify Ollama is running and accessible

---

**Enjoy using your enhanced SRM AI Doc Assist web application! 🎉**
