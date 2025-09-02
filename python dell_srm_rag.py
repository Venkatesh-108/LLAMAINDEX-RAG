#!/usr/bin/env python3
"""
Complete Dell SRM RAG CLI System
Optimized for Python 3.10 with local Ollama models
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import re
import pandas as pd
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich import print as rprint

# LlamaIndex imports
try:
    from llama_index.core import (
        VectorStoreIndex, 
        SimpleDirectoryReader, 
        Settings,
        StorageContext,
        load_index_from_storage
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode, Document
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.readers.file import PDFReader
except ImportError as e:
    print(f"❌ Missing LlamaIndex dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Other imports
try:
    import chromadb
    import pymupdf
    import ollama
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Initialize Rich console
console = Console()

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    pdf_directory: str = "./dell_srm_pdfs"
    vector_db_path: str = "./vector_db"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    similarity_top_k: int = 8
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 120

class OllamaManager:
    """Manages Ollama model operations"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.client = ollama.Client(host=host)
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception:
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            available_models = self.list_models()
            if model_name not in available_models:
                console.print(f"📥 Pulling model: {model_name}")
                with console.status(f"Downloading {model_name}..."):
                    self.client.pull(model_name)
                console.print(f"✅ Model {model_name} downloaded successfully")
            return True
        except Exception as e:
            console.print(f"❌ Failed to pull model {model_name}: {e}")
            return False

class DellSRMDocumentProcessor:
    """Specialized processor for Dell SRM documentation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.pdf_reader = PDFReader()
        self.text_splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Dell SRM specific patterns
        self.srm_patterns = {
            'chapter_headers': [
                r'^Chapter \d+',
                r'^Part \d+',
                r'^\d+\.\d+\s+[A-Z]',
            ],
            'table_indicators': [
                'Table \d+',
                'Configuration Matrix',
                'System Requirements',
                'Port Assignments',
                'Alert Definitions',
                'SolutionPack'
            ],
            'procedure_steps': [
                r'^\d+\.\s+',
                r'^Step \d+',
                r'^\w+\)\s+',
            ],
            'srm_terminology': [
                'SolutionPacks', 'Storage Monitoring', 'WWN', 'Fabric',
                'Zone', 'Host Discovery', 'Topology', 'Dell EMC',
                'SRM Console', 'Report Library', 'Infrastructure'
            ]
        }
    
    def process_pdf_directory(self) -> List[Document]:
        """Process all PDFs in the directory"""
        pdf_path = Path(self.config.pdf_directory)
        
        if not pdf_path.exists():
            console.print(f"❌ PDF directory not found: {pdf_path}")
            console.print(f"Please create directory and add your Dell SRM PDFs")
            return []
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            console.print(f"❌ No PDF files found in: {pdf_path}")
            return []
        
        console.print(f"📁 Found {len(pdf_files)} PDF files")
        
        all_documents = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            
            for pdf_file in pdf_files:
                progress.update(task, description=f"Processing {pdf_file.name}")
                
                try:
                    documents = self._process_single_pdf(pdf_file)
                    all_documents.extend(documents)
                    progress.advance(task)
                except Exception as e:
                    console.print(f"❌ Error processing {pdf_file.name}: {e}")
                    progress.advance(task)
        
        console.print(f"✅ Processed {len(all_documents)} document chunks")
        return all_documents
    
    def _process_single_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF file"""
        try:
            # Load PDF
            reader = SimpleDirectoryReader(
                input_files=[str(pdf_path)],
                file_extractor={".pdf": self.pdf_reader}
            )
            documents = reader.load_data()
            
            if not documents:
                return []
            
            # Process each document
            processed_docs = []
            for doc in documents:
                # Extract hierarchical content
                enhanced_doc = self._enhance_document_metadata(doc, pdf_path.name)
                
                # Extract tables
                table_docs = self._extract_tables_from_text(doc.text, pdf_path.name)
                
                processed_docs.append(enhanced_doc)
                processed_docs.extend(table_docs)
            
            return processed_docs
            
        except Exception as e:
            console.print(f"Error processing {pdf_path}: {e}")
            return []
    
    def _enhance_document_metadata(self, doc: Document, filename: str) -> Document:
        """Enhance document with SRM-specific metadata"""
        
        # Detect document sections
        sections = self._detect_srm_sections(doc.text)
        
        # Count SRM terminology
        srm_term_count = sum(
            doc.text.lower().count(term.lower()) 
            for term in self.srm_patterns['srm_terminology']
        )
        
        # Enhanced metadata
        doc.metadata.update({
            'filename': filename,
            'is_srm_doc': True,
            'srm_terminology_count': srm_term_count,
            'section_count': len(sections),
            'has_tables': self._has_tables(doc.text),
            'has_procedures': self._has_procedures(doc.text),
            'document_type': self._classify_document_type(doc.text, filename)
        })
        
        return doc
    
    def _extract_tables_from_text(self, text: str, filename: str) -> List[Document]:
        """Extract table-like content from text"""
        table_docs = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        table_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Detect table start
            if any(indicator in line for indicator in self.srm_patterns['table_indicators']):
                if current_table:
                    # Save previous table
                    table_doc = self._create_table_document(
                        current_table, filename, table_count
                    )
                    if table_doc:
                        table_docs.append(table_doc)
                
                current_table = [line]
                in_table = True
                table_count += 1
                continue
            
            # Collect table content
            if in_table:
                if line and (
                    '|' in line or 
                    '\t' in line or 
                    re.match(r'\s*\w+\s+\w+\s+\w+', line) or
                    re.match(r'^\w+\s*:', line)
                ):
                    current_table.append(line)
                elif len(current_table) > 3:  # End of table
                    table_doc = self._create_table_document(
                        current_table, filename, table_count
                    )
                    if table_doc:
                        table_docs.append(table_doc)
                    current_table = []
                    in_table = False
        
        # Handle last table
        if current_table and len(current_table) > 3:
            table_doc = self._create_table_document(
                current_table, filename, table_count
            )
            if table_doc:
                table_docs.append(table_doc)
        
        return table_docs
    
    def _create_table_document(self, table_lines: List[str], filename: str, table_id: int) -> Optional[Document]:
        """Create a document from table content"""
        if len(table_lines) < 2:
            return None
        
        table_text = '\n'.join(table_lines)
        
        # Create table summary
        summary = f"Table from {filename} with {len(table_lines)} rows"
        if table_lines:
            summary += f". Title: {table_lines[0][:50]}..."
        
        doc = Document(
            text=table_text,
            metadata={
                'filename': filename,
                'content_type': 'table',
                'table_id': f"{filename}_table_{table_id}",
                'table_rows': len(table_lines),
                'table_summary': summary,
                'is_srm_doc': True
            }
        )
        
        return doc
    
    def _detect_srm_sections(self, text: str) -> List[Dict]:
        """Detect SRM document sections"""
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            for pattern in self.srm_patterns['chapter_headers']:
                if re.match(pattern, line):
                    sections.append({
                        'line_number': i,
                        'title': line,
                        'type': 'header'
                    })
        
        return sections
    
    def _has_tables(self, text: str) -> bool:
        """Check if document contains tables"""
        return any(
            indicator in text 
            for indicator in self.srm_patterns['table_indicators']
        )
    
    def _has_procedures(self, text: str) -> bool:
        """Check if document contains procedures"""
        return any(
            re.search(pattern, text) 
            for pattern in self.srm_patterns['procedure_steps']
        )
    
    def _classify_document_type(self, text: str, filename: str) -> str:
        """Classify the type of SRM document"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if 'admin' in filename_lower or 'administrator' in text_lower:
            return 'admin_guide'
        elif 'user' in filename_lower or 'user guide' in text_lower:
            return 'user_guide'
        elif 'install' in filename_lower or 'installation' in text_lower:
            return 'installation_guide'
        elif 'sp' in filename_lower or 'solutionpack' in text_lower:
            return 'solutionpack_guide'
        else:
            return 'general_guide'

class RAGVectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.index = None
        
        # Setup storage directory
        self.storage_dir = Path(config.vector_db_path)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.storage_dir / "chroma_db")
        )
    
    def create_or_load_index(self, documents: List[Document] = None) -> bool:
        """Create new index or load existing one"""
        
        # Check if index exists
        index_path = self.storage_dir / "index"
        
        if index_path.exists() and not documents:
            return self._load_existing_index()
        else:
            return self._create_new_index(documents)
    
    def _load_existing_index(self) -> bool:
        """Load existing index"""
        try:
            console.print("📂 Loading existing vector index...")
            
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_dir / "index")
            )
            
            self.index = load_index_from_storage(storage_context)
            console.print("✅ Vector index loaded successfully")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to load existing index: {e}")
            return False
    
    def _create_new_index(self, documents: List[Document]) -> bool:
        """Create new vector index"""
        if not documents:
            console.print("❌ No documents provided for indexing")
            return False
        
        try:
            console.print("🔨 Creating new vector index...")
            
            # Setup ChromaDB collection
            collection_name = "dell_srm_docs"
            try:
                # Try to delete existing collection
                self.chroma_client.delete_collection(collection_name)
            except:
                pass  # Collection doesn't exist
            
            chroma_collection = self.chroma_client.create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            with console.status("Building vector index..."):
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=False
                )
            
            # Persist index
            self.index.storage_context.persist(
                persist_dir=str(self.storage_dir / "index")
            )
            
            console.print(f"✅ Vector index created with {len(documents)} documents")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to create vector index: {e}")
            return False
    
    def get_query_engine(self):
        """Get query engine for the index"""
        if not self.index:
            return None
        
        return self.index.as_query_engine(
            similarity_top_k=self.config.similarity_top_k,
            response_mode="compact",
            streaming=False
        )

class DellSRMRAG:
    """Main RAG system for Dell SRM documents"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.ollama_manager = OllamaManager(self.config.ollama_host)
        self.doc_processor = DellSRMDocumentProcessor(self.config)
        self.vector_store = RAGVectorStore(self.config)
        self.query_engine = None
        self.setup_logging()
        
        # Setup LlamaIndex global settings
        Settings.llm = Ollama(
            model=self.config.llm_model,
            base_url=self.config.ollama_host,
            temperature=self.config.temperature,
            request_timeout=self.config.timeout
        )
        
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config.embedding_model,
            base_url=self.config.ollama_host
        )
    
    def _load_config(self, config_path: Optional[str]) -> RAGConfig:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                return RAGConfig(**config_data)
            except Exception as e:
                console.print(f"⚠️ Failed to load config: {e}. Using defaults.")
        
        return RAGConfig()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dell_srm_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the RAG system"""
        console.print("\n🚀 Initializing Dell SRM RAG System")
        console.print("=" * 50)
        
        # Check Ollama
        if not self._check_ollama_setup():
            return False
        
        # Check for existing index or process documents
        if not self._setup_vector_index():
            return False
        
        # Create query engine
        self.query_engine = self.vector_store.get_query_engine()
        if not self.query_engine:
            console.print("❌ Failed to create query engine")
            return False
        
        console.print("✅ RAG System initialized successfully!")
        return True
    
    def _check_ollama_setup(self) -> bool:
        """Check and setup Ollama"""
        console.print("🔍 Checking Ollama setup...")
        
        # Check if Ollama is running
        if not self.ollama_manager.check_ollama_running():
            console.print("❌ Ollama is not running")
            console.print("Please start Ollama: ollama serve")
            return False
        
        console.print("✅ Ollama is running")
        
        # Check and pull required models
        required_models = [self.config.llm_model, self.config.embedding_model]
        available_models = self.ollama_manager.list_models()
        
        for model in required_models:
            if model not in available_models:
                if not self.ollama_manager.pull_model(model):
                    return False
        
        console.print(f"✅ Required models available: {required_models}")
        return True
    
    def _setup_vector_index(self) -> bool:
        """Setup vector index"""
        console.print("📊 Setting up vector index...")
        
        # Check if we should rebuild
        rebuild = False
        if not (Path(self.config.vector_db_path) / "index").exists():
            rebuild = True
            console.print("No existing index found, will create new one")
        
        documents = None
        if rebuild:
            # Process documents
            documents = self.doc_processor.process_pdf_directory()
            if not documents:
                console.print("❌ No documents to process")
                return False
        
        # Create or load index
        return self.vector_store.create_or_load_index(documents)
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        if not self.query_engine:
            return {"error": "RAG system not initialized"}
        
        try:
            # Enhanced prompt for SRM context
            enhanced_prompt = self._create_enhanced_prompt(question)
            
            # Execute query
            start_time = time.time()
            response = self.query_engine.query(enhanced_prompt)
            query_time = time.time() - start_time
            
            # Process response
            result = {
                "answer": str(response),
                "query_time": round(query_time, 2),
                "sources": []
            }
            
            # Extract source information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes[:5]):  # Top 5 sources
                    source_info = {
                        "source_id": i + 1,
                        "filename": node.metadata.get('filename', 'Unknown'),
                        "content_type": node.metadata.get('content_type', 'text'),
                        "document_type": node.metadata.get('document_type', 'general'),
                        "similarity_score": getattr(node, 'score', 0.0),
                        "excerpt": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    result["sources"].append(source_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {"error": f"Query failed: {str(e)}"}
    
    def _create_enhanced_prompt(self, question: str) -> str:
        """Create enhanced prompt for Dell SRM context"""
        prompt_template = """
You are an expert assistant for Dell SRM (Storage Resource Management) documentation.

When answering questions about Dell SRM:

1. FOCUS ON ACCURACY: Only provide information found in the Dell SRM documentation
2. BE SPECIFIC: Reference specific features, procedures, and configurations
3. USE TECHNICAL TERMS: Employ proper Dell SRM terminology (SolutionPacks, WWN, Fabric, etc.)
4. PROVIDE CONTEXT: Include relevant background information when helpful
5. STRUCTURE CLEARLY: Organize complex answers with clear sections
6. FLAG LIMITATIONS: State if information is incomplete or requires additional context

Dell SRM Key Areas:
- Installation and System Requirements
- SolutionPack Configuration
- Storage Monitoring and Reporting  
- Host Discovery and Fabric Management
- Alert Configuration and Troubleshooting
- User Interface and Navigation

User Question: {question}

Provide a comprehensive, accurate answer based on the Dell SRM documentation:
        """.strip()
        
        return prompt_template.format(question=question)
    
    def interactive_mode(self):
        """Run interactive query mode"""
        console.print("\n💬 Dell SRM RAG Interactive Mode")
        console.print("Ask questions about Dell SRM documentation")
        console.print("Type 'quit', 'exit', or 'q' to stop")
        console.print("=" * 50)
        
        while True:
            try:
                question = Prompt.ask("\n🤔 Your question")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                # Execute query
                with console.status("🔍 Searching Dell SRM documentation..."):
                    result = self.query(question)
                
                if "error" in result:
                    console.print(f"❌ Error: {result['error']}")
                    continue
                
                # Display results
                self._display_query_result(result)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"❌ Unexpected error: {e}")
    
    def _display_query_result(self, result: Dict):
        """Display query results in a formatted way"""
        
        # Display answer
        console.print("\n📋 Answer:")
        answer_panel = Panel(
            result["answer"],
            title="Dell SRM Assistant Response",
            border_style="green"
        )
        console.print(answer_panel)
        
        # Display sources if available
        if result.get("sources"):
            console.print(f"\n📚 Sources ({len(result['sources'])} found):")
            
            sources_table = Table(show_header=True, header_style="bold blue")
            sources_table.add_column("#", style="dim", width=3)
            sources_table.add_column("Document", style="cyan")
            sources_table.add_column("Type", style="magenta")
            sources_table.add_column("Relevance", style="green")
            sources_table.add_column("Preview", style="dim")
            
            for source in result["sources"]:
                relevance = f"{source['similarity_score']:.1%}" if source['similarity_score'] else "N/A"
                preview = source["excerpt"][:80] + "..." if len(source["excerpt"]) > 80 else source["excerpt"]
                
                sources_table.add_row(
                    str(source["source_id"]),
                    source["filename"],
                    source["content_type"],
                    relevance,
                    preview
                )
            
            console.print(sources_table)
        
        # Display query time
        if result.get("query_time"):
            console.print(f"\n⏱️  Query time: {result['query_time']}s")
    
    def save_config(self, path: str):
        """Save current configuration"""
        config_dict = asdict(self.config)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        console.print(f"✅ Configuration saved to {path}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Dell SRM RAG System - AI Assistant for Dell SRM Documentation"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--pdf-dir", "-p",
        type=str,
        default="./dell_srm_pdfs",
        help="Directory containing Dell SRM PDF files"
    )
    
    parser.add_argument(
        "--rebuild-index", "-r",
        action="store_true",
        help="Force rebuild of vector index"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode (non-interactive)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.1:8b",
        help="LLM model to use"
    )
    
    args = parser.parse_args()
    
    # Welcome message
    console.print("\n🏢 Dell SRM RAG System")
    console.print("AI-Powered Assistant for Dell SRM Documentation")
    console.print("=" * 60)
    
    try:
        # Initialize RAG system
        rag = DellSRMRAG(args.config)
        
        # Update config from args
        if args.pdf_dir:
            rag.config.pdf_directory = args.pdf_dir
        if args.model:
            rag.config.llm_model = args.model
        
        # Force rebuild if requested
        if args.rebuild_index:
            import shutil
            if Path(rag.config.vector_db_path).exists():
                shutil.rmtree(rag.config.vector_db_path)
                console.print("🗑️ Removed existing vector index")
        
        # Initialize system
        if not rag.initialize():
            console.print("❌ Failed to initialize RAG system")
            return 1
        
        # Single query mode
        if args.query:
            with console.status("🔍 Processing query..."):
                result = rag.query(args.query)
            
            if "error" in result:
                console.print(f"❌ Error: {result['error']}")
                return 1
            
            rag._display_query_result(result)
            return 0
        
        # Interactive mode
        rag.interactive_mode()
        return 0
        
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        console.print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
