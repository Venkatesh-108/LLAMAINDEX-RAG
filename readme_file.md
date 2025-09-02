# Dell SRM RAG System

🏢 **AI-Powered Assistant for Dell SRM Documentation**

A complete Retrieval-Augmented Generation (RAG) system specifically designed for Dell Storage Resource Management (SRM) documentation. This system can intelligently answer questions about Dell SRM installation, configuration, monitoring, and troubleshooting using your local PDF manuals.

## ✨ Features

- **🔍 Intelligent Document Search**: Advanced semantic search through Dell SRM documentation
- **📊 Table-Aware Processing**: Extracts and understands complex tables and configuration matrices
- **🏗️ Hierarchical Understanding**: Maintains document structure and cross-references
- **🦙 Local LLM Support**: Runs entirely on local Ollama models (no external API calls)
- **💬 Interactive CLI**: User-friendly command-line interface with rich formatting
- **⚡ Fast Retrieval**: Optimized vector search with hybrid retrieval strategies
- **🔧 Enterprise Ready**: Handles complex technical documentation with specialized processing

## 🎯 Perfect For

- **Dell SRM Administrators**: Quick access to installation and configuration procedures
- **Storage Engineers**: Technical reference for troubleshooting and optimization
- **IT Teams**: Centralized knowledge base for Dell SRM environments
- **Documentation Teams**: Enhanced searchability of technical manuals

## 📋 Requirements

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Ollama** (for local LLM inference)
- **8GB RAM minimum** (16GB recommended for better performance)
- **5GB disk space** (for models and vector database)

### Supported Operating Systems
- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- ✅ **macOS** (10.15+, Intel and Apple Silicon)
- ✅ **Windows** (10/11 with WSL2 recommended)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone or download the system files
git clone <repository-url>  # or download and extract
cd dell-srm-rag

# 2. Run automated setup
python setup.py

# 3. Add your Dell SRM PDF files
cp /path/to/your/srm-pdfs/* ./dell_srm_pdfs/

# 4. Start the system
./start_rag.sh  # Linux/macOS
# or
start_rag.bat   # Windows
```

### Option 2: Manual Setup

```bash
# 1. Install Python dependencies
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt

# 2. Install and start Ollama
# Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. Create directories and add PDFs
mkdir dell_srm_pdfs vector_db logs config
cp /path/to/your/srm-pdfs/* ./dell_srm_pdfs/

# 4. Run the system
python dell_srm_rag.py
```

## 💡 Usage Examples

### Interactive Mode (Default)
```bash
python dell_srm_rag.py

# Example queries:
🤔 Your question: What are the system requirements for Dell SRM installation?
🤔 Your question: How do I configure SolutionPacks for EMC storage?
🤔 Your question: What ports need to be open for SRM fabric discovery?
🤔 Your question: How do I troubleshoot host discovery issues?
```

### Single Query Mode
```bash
python dell_srm_rag.py --query "What are the Dell SRM installation prerequisites?"
```

### Custom Configuration
```bash
python dell_srm_rag.py --config config/custom_config.json --model llama3.1:70b
```

### Force Index Rebuild
```bash
python dell_srm_rag.py --rebuild-index --pdf-dir ./new_srm_docs
```

## 📁 Project Structure

```
dell-srm-rag/
├── dell_srm_rag.py          # Main RAG system
├── setup.py                 # Automated setup script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── start_rag.sh            # Linux/macOS start script
├── start_rag.bat           # Windows start script
├── dell_srm_pdfs/          # Place your PDF files here
├── vector_db/              # Vector database storage
├── logs/                   # System logs
└── config/                 # Configuration files
    └── config.json         # Example configuration
```

## ⚙️ Configuration

The system uses `config/config.json` for configuration:

```json
{
  "pdf_directory": "./dell_srm_pdfs",
  "vector_db_path": "./vector_db",
  "ollama_host": "http://localhost:11434",
  "llm_model": "llama3.1:8b",
  "embedding_model": "nomic-embed-text",
  "chunk_size": 1024,
  "chunk_overlap": 128,
  "similarity_top_k": 8,
  "temperature": 0.1,
  "max_retries": 3,
  "timeout": 120
}
```

### Available Models

**LLM Models (choose based on your hardware):**
- `llama3.1:8b` - Good balance of speed and quality (8GB RAM)
- `llama3.1:70b` - Higher quality but slower (32GB+ RAM)
- `qwen2.5:14b` - Excellent for technical documents (16GB RAM)
- `mistral-nemo:12b` - Fast and efficient (12GB RAM)

**Embedding Models:**
- `nomic-embed-text` - Optimized for retrieval (recommended)
- `mxbai-embed-large` - Higher quality embeddings

## 🔧 Command Line Options

```bash
python dell_srm_rag.py [OPTIONS]

Options:
  -c, --config PATH           Configuration file path
  -p, --pdf-dir PATH         PDF directory path
  -r, --rebuild-index        Force rebuild of vector index
  -q, --query TEXT           Single query mode (non-interactive)
  -m, --model TEXT           LLM model to use
  --help                     Show help message
```

## 📊 Sample Queries for Dell SRM

### Installation & Setup
- "What are the system requirements for Dell SRM 4.4?"
- "How do I install Dell SRM on Windows Server?"
- "What database requirements does SRM have?"

### Configuration
- "How do I configure SolutionPacks for EMC storage arrays?"
- "What are the steps to set up fabric discovery?"
- "How do I configure SNMP monitoring in SRM?"

### Monitoring & Reporting
- "How do I create custom reports in Dell SRM?"
- "What alerts can I configure for storage monitoring?"
- "How do I set up automated report scheduling?"

### Troubleshooting
- "Why is host discovery not working in my fabric?"
- "How do I troubleshoot SolutionPack connection issues?"
- "What do I do if SRM performance is slow?"

## 🎛️ Advanced Features

### Document Type Recognition
The system automatically recognizes different types of Dell SRM documents:
- **Admin Guides**: Installation, configuration, administration
- **User Guides**: End-user operations and reporting
- **SP Guides**: SolutionPack-specific documentation
- **Installation Guides**: Setup and deployment procedures

### Intelligent Table Processing
- Extracts configuration matrices and system requirements
- Preserves table structure and relationships
- Provides table-aware search and retrieval

### Hierarchical Context
- Maintains document section relationships
- Provides parent-child context in responses
- Cross-references related sections automatically

## 🚨 Troubleshooting

### Common Issues

**1. "Ollama is not running"**
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434
```

**2. "No PDF files found"**
```bash
# Check PDF directory
ls -la dell_srm_pdfs/

# Add PDF files
cp /path/to/srm-pdfs/*.pdf dell_srm_pdfs/
```

**3. "Failed to create vector index"**
```bash
# Clear existing index and rebuild
rm -rf vector_db/
python dell_srm_rag.py --rebuild-index
```

**4. "Model not found"**
```bash
# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# List available models
ollama list
```

**5. Memory Issues**
- Use smaller models: `llama3.1:8b` instead of `llama3.1:70b`
- Reduce `chunk_size` and `similarity_top_k` in config
- Close other applications to free RAM

### Performance Optimization

**For Better Speed:**
- Use SSD storage for vector database
- Increase RAM allocation
- Use smaller, faster models

**For Better Quality:**
- Use larger models (`llama3.1:70b`)
- Increase `similarity_top_k` for more context
- Fine-tune chunk size for your document types

## 📈 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dell SRM RAG System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   Dell SRM      │    │   Document       │    │   Vector    │ │
│  │   PDF Manuals   │───►│   Processing     │───►│   Database  │ │
│  │   (Input)       │    │   (LlamaIndex)   │    │  (ChromaDB) │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   User Query    │    │   Hybrid         │    │   LLM       │ │
│  │   (CLI)         │◄───│   Retrieval      │◄───│   Ollama    │ │
│  │                 │    │   Engine         │    │   (Local)   │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LlamaIndex** - Excellent RAG framework
- **Ollama** - Local LLM inference platform  
- **ChromaDB** - Vector database
- **Rich** - Beautiful CLI formatting

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the logs in `logs/dell_srm_rag.log`
3. Open an issue with detailed error information
4. Include your system specs and configuration

---

**Happy querying your Dell SRM documentation!** 🚀