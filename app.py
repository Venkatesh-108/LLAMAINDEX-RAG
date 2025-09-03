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
    
    # Advanced retrieval enhancements (optional)
    try:
        from llama_index.core.retrievers import (
            VectorIndexRetriever
        )
        from llama_index.core.postprocessor import (
            SimilarityPostprocessor,
            LLMRerank
        )
        from llama_index.core.query_engine import RetrieverQueryEngine
        
        # Check for additional advanced features
        try:
            from llama_index.core.retrievers import HybridRetriever
            HYBRID_AVAILABLE = True
        except ImportError:
            HYBRID_AVAILABLE = False
            
        try:
            from llama_index.core.retrievers import MultiQueryRetriever
            MULTI_QUERY_AVAILABLE = True
        except ImportError:
            MULTI_QUERY_AVAILABLE = False
            
        try:
            from llama_index.core.retrievers import BM25Retriever
            BM25_AVAILABLE = True
        except ImportError:
            BM25_AVAILABLE = False
        
        ADVANCED_RETRIEVAL_AVAILABLE = True
        
    except ImportError as e:
        ADVANCED_RETRIEVAL_AVAILABLE = False
        HYBRID_AVAILABLE = False
        MULTI_QUERY_AVAILABLE = False
        BM25_AVAILABLE = False
        
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

# Now we can use console.print for advanced retrieval status
if ADVANCED_RETRIEVAL_AVAILABLE:
    available_features = []
    if HYBRID_AVAILABLE:
        available_features.append("Hybrid Retrieval")
    if MULTI_QUERY_AVAILABLE:
        available_features.append("Multi-Query")
    if BM25_AVAILABLE:
        available_features.append("BM25 Sparse Retrieval")
    
    if available_features:
        console.print(f"✅ Advanced retrieval features available: {', '.join(available_features)}")
    else:
        console.print("✅ Basic advanced features available (SimilarityPostprocessor, LLMRerank)")
else:
    console.print("⚠️ Advanced retrieval features not available - using basic retrieval")

@dataclass
class RAGConfig:
    """Configuration for the RAG system - Optimized for accuracy"""
    pdf_directory: str = "./documents"
    vector_db_path: str = "./vector_db"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    # Enhanced chunking for better context preservation
    chunk_size: int = 1024  # Optimal size for technical documentation
    chunk_overlap: int = 200  # Increased overlap for better continuity
    similarity_top_k: int = 12  # More candidates for better selection
    temperature: float = 0.1  # Low temperature for accurate responses
    max_retries: int = 3
    timeout: int = 120
    # New accuracy parameters
    rerank_threshold: float = 0.3  # Similarity threshold for reranking
    max_context_length: int = 4096  # Maximum context window
    use_query_enhancement: bool = True  # New setting to control enhancement

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
            chunk_overlap=config.chunk_overlap,
            separator="\\n\\n"  # Split by paragraphs for better semantic coherence
        )
        
        # Dell SRM specific patterns
        self.srm_patterns = {
            'chapter_headers': [
                r'^Chapter \d+',
                r'^Part \d+',
                r'^\d+\.\d+\s+[A-Z]',
            ],
            'table_indicators': [
                r'Table \d+',
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

class ContextFilter:
    """Filters and ranks passages based on relevance and quality"""
    
    def __init__(self):
        self.quality_indicators = [
            "step", "procedure", "configuration", "requirement",
            "installation", "troubleshoot", "error", "solution",
            "note:", "warning:", "important:"
        ]

    def filter_and_rank(self, query: str, passages: List) -> List:
        """Filter and rank passages"""
        
        filtered_passages = []
        query_lower = query.lower()
        
        for passage in passages:
            passage_text = getattr(passage, 'text', '')
            passage_score = getattr(passage, 'score', 0.0)
            
            # Calculate quality score
            quality_score = self._calculate_quality(passage_text, query_lower)
            
            # Combine original score with quality score
            combined_score = (passage_score * 0.6) + (quality_score * 0.4)
            
            # Add combined score to passage metadata for ranking
            if hasattr(passage, 'metadata'):
                passage.metadata['combined_score'] = combined_score
            else:
                # If no metadata, create a simple object to hold it
                passage.combined_score = combined_score
            
            filtered_passages.append(passage)
            
        # Sort by the new combined score
        return sorted(
            filtered_passages, 
            key=lambda p: getattr(p, 'metadata', {}).get('combined_score', getattr(p, 'combined_score', 0.0)), 
            reverse=True
        )

    def _calculate_quality(self, text: str, query: str) -> float:
        """Calculate quality score for a passage"""
        text_lower = text.lower()
        
        # Technical content indicators
        technical_score = sum(
            1 for indicator in self.quality_indicators 
            if indicator in text_lower
        ) / len(self.quality_indicators)
        
        # Query term overlap
        query_terms = set(query.split())
        text_terms = set(text_lower.split())
        
        if not query_terms:
            overlap = 0.0
        else:
            overlap = len(query_terms.intersection(text_terms)) / len(query_terms)
        
        # Content length (prefer substantial content)
        length_score = min(len(text.split()) / 150, 1.0) # Prefer chunks of at least 150 words
        
        # Final weighted score
        return (technical_score * 0.45 + overlap * 0.35 + length_score * 0.2)


class EnhancedRetriever:
    """Enhanced retrieval techniques compatible with current LlamaIndex version"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = None
    
    def setup_llm(self, llm):
        """Setup LLM for enhanced features"""
        self.llm = llm
    
    def enhance_query(self, query: str) -> str:
        """Enhance query with Dell SRM context and synonyms"""
        srm_synonyms = {
            "storage": ["storage array", "storage system", "EMC", "Dell EMC", "storage infrastructure"],
            "monitoring": ["monitor", "surveillance", "tracking", "alerting", "observability"],
            "solutionpack": ["SP", "Solution Pack", "monitoring pack", "SolutionPack"],
            "fabric": ["SAN fabric", "storage fabric", "network fabric", "Fibre Channel fabric"],
            "WWN": ["World Wide Name", "port identifier", "FC port ID"],
            "host": ["server", "initiator", "host system", "compute node"],
            "array": ["storage array", "disk array", "storage system", "EMC array"],
            "zone": ["zoning", "SAN zone", "fabric zone", "storage zone"],
            "alert": ["notification", "alarm", "warning", "event"],
            "report": ["reporting", "analytics", "dashboard", "metrics"]
        }
        
        enhanced_query = query
        query_lower = query.lower()
        
        for term, synonyms in srm_synonyms.items():
            if term.lower() in query_lower:
                # Add synonyms that aren't already in the query
                for synonym in synonyms:
                    if synonym.lower() not in query_lower:
                        enhanced_query += f" {synonym}"
        
        # Add Dell SRM context if not present
        if "dell srm" not in query_lower and "srm" not in query_lower:
            enhanced_query += " Dell SRM Storage Resource Management"
        
        return enhanced_query.strip()
    
    def create_enhanced_retriever(self, index):
        """Create enhanced retriever with better parameters"""
        try:
            # Use advanced retriever if available
            if ADVANCED_RETRIEVAL_AVAILABLE:
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=self.config.similarity_top_k * 2  # Get more candidates
                )
            else:
                # Fallback to basic retriever
                retriever = index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k * 2
                )
            
            return retriever
            
        except Exception as e:
            console.print(f"⚠️ Enhanced retriever failed, using basic: {e}")
            return index.as_retriever(
                similarity_top_k=self.config.similarity_top_k
            )


class RAGVectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.index = None
        self.enhanced_retriever = None
        
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
            success = self._load_existing_index()
        else:
            success = self._create_new_index(documents)
        
        # Initialize enhanced retriever if index is available
        if success and self.index:
            self.enhanced_retriever = EnhancedRetriever(self.config)
        
        return success
    
    def _load_existing_index(self) -> bool:
        """Load existing index"""
        try:
            console.print("📂 Loading existing vector index...")

            # Connect to existing ChromaDB collection
            collection_name = "dell_srm_docs"
            try:
                chroma_collection = self.chroma_client.get_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_dir / "index"),
                    vector_store=vector_store
                )
            except Exception:
                # Fallback to default if collection doesn't exist
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
        """Get enhanced query engine for the index"""
        if not self.index:
            return None

        try:
            # Use enhanced retriever if available
            if self.enhanced_retriever:
                base_retriever = self.enhanced_retriever.create_enhanced_retriever(self.index)
            else:
                base_retriever = self.index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k * 2
                )

            # Use available advanced features
            if ADVANCED_RETRIEVAL_AVAILABLE:
                try:
                    # Post-processor to filter low-similarity results
                    similarity_filter = SimilarityPostprocessor(
                        similarity_cutoff=self.config.rerank_threshold
                    )

                    # Create enhanced query engine with available features
                    query_engine = RetrieverQueryEngine.from_args(
                        retriever=base_retriever,
                        node_postprocessors=[similarity_filter],
                        response_mode="tree_summarize",  # Better for technical documentation
                        streaming=True, # Enable streaming
                        verbose=True  # Enable verbose mode for debugging
                    )

                    # Show enhanced features status
                    if self.enhanced_retriever:
                        console.print("✅ Enhanced retrieval features enabled:")
                        console.print(f"   - Query enhancement: ✅")
                        console.print(f"   - Advanced retriever: ✅")
                        console.print(f"   - Similarity filtering: ✅")
                        
                        # Show available advanced features
                        if HYBRID_AVAILABLE:
                            console.print(f"   - Hybrid retrieval: ✅")
                        if MULTI_QUERY_AVAILABLE:
                            console.print(f"   - Multi-query: ✅")
                        if BM25_AVAILABLE:
                            console.print(f"   - BM25 sparse retrieval: ✅")
                    else:
                        console.print("⚠️ Using basic retrieval (enhanced features not available)")

                    return query_engine

                except Exception as e:
                    console.print(f"⚠️ Advanced features failed, using basic: {e}")
                    # Fallback to basic query engine
                    if hasattr(base_retriever, 'as_query_engine'):
                        return base_retriever.as_query_engine(
                            response_mode="tree_summarize",
                            use_async=True,
                            verbose=True
                        )
                    else:
                        # Use the index directly
                        return self.index.as_query_engine(
                            response_mode="tree_summarize",
                            use_async=True,
                            verbose=True
                        )
            else:
                # Use basic query engine
                console.print("⚠️ Advanced features not available, using basic query engine")
                if hasattr(base_retriever, 'as_query_engine'):
                    return base_retriever.as_query_engine(
                        response_mode="tree_summarize",
                        use_async=True,
                        verbose=True
                    )
                else:
                    # Use the index directly
                    return self.index.as_query_engine(
                        response_mode="tree_summarize",
                        use_async=True,
                        verbose=True
                    )

        except Exception as e:
            console.print(f"⚠️ Enhanced query engine failed, falling back to basic: {e}")
            # Fallback to basic retriever
            base_retriever = self.index.as_retriever(
                similarity_top_k=self.config.similarity_top_k
            )
            
            # Use the index directly for query engine
            return self.index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                verbose=True
            )

class DellSRMRAG:
    """Main RAG system for Dell SRM documents"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.ollama_manager = OllamaManager(self.config.ollama_host)
        self.doc_processor = DellSRMDocumentProcessor(self.config)
        self.vector_store = RAGVectorStore(self.config)
        self.context_filter = ContextFilter()  # Add context filter
        self.query_engine = None
        self.setup_logging()
        
        # Setup LlamaIndex global settings
        # Enhanced LLM settings for better accuracy
        Settings.llm = Ollama(
            model=self.config.llm_model,
            base_url=self.config.ollama_host,
            temperature=0.1,  # Lower temperature for more accurate responses
            request_timeout=self.config.timeout,
            context_window=4096,  # Increase context window if model supports it
            additional_kwargs={
                "top_p": 0.1,  # More focused responses
                "top_k": 40,   # Consider top 40 tokens
                "num_predict": 512  # Reasonable response length
            }
        )

        # Enhanced embedding settings
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config.embedding_model,
            base_url=self.config.ollama_host,
            embed_batch_size=10,  # Process in smaller batches for accuracy
            query_instruction="Represent the question for retrieving relevant technical documentation about Dell SRM systems:",
            text_instruction="Represent the technical documentation chunk for retrieval:"
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
        
        # Setup enhanced retriever with LLM
        if self.vector_store.enhanced_retriever:
            self.vector_store.enhanced_retriever.setup_llm(Settings.llm)
        
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
            # Enhanced query processing
            enhanced_query = self._enhance_query(question)
            
            # Use enhanced retriever for additional query enhancement if available
            if self.vector_store.enhanced_retriever:
                enhanced_query = self.vector_store.enhanced_retriever.enhance_query(enhanced_query)

            # Create enhanced prompt for SRM context
            enhanced_prompt = self._create_enhanced_prompt(enhanced_query)

            # Execute query with enhanced retrieval
            start_time = time.time()
            response = self.query_engine.query(enhanced_prompt)
            query_time = time.time() - start_time
            
            # Check if we have sufficient source data
            has_sufficient_data = False
            relevant_sources = []

            if hasattr(response, 'source_nodes') and response.source_nodes:
                # Filter and rank sources
                ranked_sources = self.context_filter.filter_and_rank(
                    enhanced_query, response.source_nodes
                )
                
                # Filter sources that meet minimum similarity threshold
                min_similarity = 0.3  # Minimum similarity threshold
                for node in ranked_sources:
                    similarity_score = getattr(node, 'score', 0.0)
                    if similarity_score >= min_similarity:
                        relevant_sources.append(node)

                # Only consider response valid if we have at least 2 relevant sources
                # or 1 source with high confidence (>0.6)
                if len(relevant_sources) >= 2 or (len(relevant_sources) == 1 and getattr(relevant_sources[0], 'score', 0.0) > 0.6):
                    has_sufficient_data = True

            # If no sufficient data found, return "no data available" message
            if not has_sufficient_data:
                return {
                    "answer": "I apologize, but I don't have sufficient information about this topic in the Dell SRM documentation. The available documents don't contain relevant data for your question.",
                    "query_time": round(query_time, 2),
                    "sources": [],
                    "data_available": False
                }

            # Process response only if we have sufficient data
            result = {
                "answer": str(response),
                "query_time": round(query_time, 2),
                "sources": [],
                "data_available": True,
                "performance": {
                    "total_time": round(query_time, 2),
                    # Other metrics would be added here if we were tracking them in non-streaming mode
                }
            }

            # Extract source information for relevant sources only
            for i, node in enumerate(relevant_sources[:5]):  # Top 5 relevant sources
                text = node.text.strip().replace('\\n', ' ')
                is_truncated = len(text) > 200
                excerpt = (text[:200] + '...') if is_truncated else text

                source_info = {
                    "source_id": i + 1,
                    "filename": node.metadata.get('filename', 'Unknown'),
                    "page_number": node.metadata.get('page_label', None),
                    "content_type": node.metadata.get('content_type', 'text'),
                    "document_type": node.metadata.get('document_type', 'general'),
                    "similarity_score": getattr(node, 'score', 0.0),
                    "excerpt": excerpt,
                    "full_text": node.text,
                    "is_truncated": is_truncated
                }
                result["sources"].append(source_info)

            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {"error": f"Query failed: {str(e)}"}
    
    def stream_query(self, question: str):
        """Streaming query the RAG system"""
        if not self.query_engine:
            yield {"type": "error", "content": "RAG system not initialized"}
            return

        try:
            import time
            start_time = time.time()
            
            # Track query enhancement time
            enhancement_start = time.time()
            enhanced_query = self._enhance_query(question)
            enhancement_time = time.time() - enhancement_start
            
            # Track prompt creation time
            prompt_start = time.time()
            enhanced_prompt = self._create_enhanced_prompt(enhanced_query)
            prompt_time = time.time() - prompt_start
            
            # Track LLM response time
            llm_start = time.time()
            streaming_response = self.query_engine.query(enhanced_prompt)
            
            # Stream tokens and count them
            token_count = 0
            for token in streaming_response.response_gen:
                token_count += 1
                yield {"type": "token", "content": token}
            
            llm_time = time.time() - llm_start
            
            # Track source processing time
            source_start = time.time()
            source_nodes = streaming_response.source_nodes
            relevant_sources = []
            if source_nodes:
                ranked_sources = self.context_filter.filter_and_rank(enhanced_query, source_nodes)
                min_similarity = 0.3
                for node in ranked_sources:
                    if getattr(node, 'score', 0.0) >= min_similarity:
                        relevant_sources.append(node)
            source_time = time.time() - source_start
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Prepare performance metrics
            performance_metrics = {
                "total_time": round(total_time, 3),
                "enhancement_time": round(enhancement_time, 3),
                "prompt_time": round(prompt_time, 3),
                "llm_time": round(llm_time, 3),
                "source_time": round(source_time, 3),
                "token_count": token_count,
                "tokens_per_second": round(token_count / llm_time if llm_time > 0 else 0, 2),
                "sources_found": len(source_nodes) if source_nodes else 0,
                "relevant_sources": len(relevant_sources),
                "model_used": self.config.llm_model,
                "efficiency_score": round((llm_time / total_time) * 100, 1) if total_time > 0 else 0,
                "enhancement_overhead": round((enhancement_time / total_time) * 100, 1) if total_time > 0 else 0,
                "source_quality": round((len(relevant_sources) / len(source_nodes) * 100) if source_nodes else 0, 1)
            }
            
            # Log detailed performance metrics
            self.logger.info("=" * 60)
            self.logger.info("🚀 PERFORMANCE METRICS SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 Total Response Time: {performance_metrics['total_time']}s")
            self.logger.info(f"🔍 Query Enhancement: {performance_metrics['enhancement_time']}s")
            self.logger.info(f"📝 Prompt Creation: {performance_metrics['prompt_time']}s")
            self.logger.info(f"🤖 LLM Generation: {performance_metrics['llm_time']}s")
            self.logger.info(f"📚 Source Processing: {performance_metrics['source_time']}s")
            self.logger.info(f"🔢 Tokens Generated: {performance_metrics['token_count']}")
            self.logger.info(f"⚡ Generation Speed: {performance_metrics['tokens_per_second']} tokens/sec")
            self.logger.info(f"📖 Sources Found: {performance_metrics['sources_found']} total, {performance_metrics['relevant_sources']} relevant")
            self.logger.info(f"⚡ Efficiency Score: {performance_metrics['efficiency_score']}% (LLM time vs total time)")
            self.logger.info(f"🎯 Source Quality: {performance_metrics['source_quality']}% (relevant vs total sources)")
            self.logger.info(f"🔍 Enhancement Overhead: {performance_metrics['enhancement_overhead']}% of total time")
            self.logger.info(f"🤖 Model Used: {performance_metrics['model_used']}")
            self.logger.info("=" * 60)
            
            sources_data = []
            for i, node in enumerate(relevant_sources[:5]):
                text = node.text.strip().replace('\\n', ' ')
                is_truncated = len(text) > 200
                excerpt = (text[:200] + '...') if is_truncated else text

                sources_data.append({
                    "source_id": i + 1,
                    "filename": node.metadata.get('filename', 'Unknown'),
                    "page_number": node.metadata.get('page_label', None),
                    "content_type": node.metadata.get('content_type', 'text'),
                    "document_type": node.metadata.get('document_type', 'general'),
                    "similarity_score": getattr(node, 'score', 0.0),
                    "excerpt": excerpt,
                    "full_text": node.text,
                    "is_truncated": is_truncated
                })
            
            yield {"type": "end", "sources": sources_data, "performance": performance_metrics}
            
        except Exception as e:
            self.logger.error(f"Streaming query failed: {e}")
            yield {"type": "error", "content": f"Streaming query failed: {str(e)}"}

    def _enhance_query(self, question: str) -> str:
        """Enhance query for better retrieval accuracy using the LLM"""
        
        if not self.config.use_query_enhancement:
            self.logger.info("Skipping LLM-based query enhancement as per configuration.")
            return question

        try:
            prompt = f"""
            Analyze the following user query about Dell SRM. Generate 3-5 alternative queries
            that are more specific, include technical synonyms, and are optimized for
            vector search in technical documentation.

            Original Query: "{question}"

            Return ONLY the alternative queries, separated by newlines.
            Example format:
            enhanced query 1
            enhanced query 2
            enhanced query 3
            """.strip()
            
            # Use a smaller, faster model for this task if available
            enhancement_llm = Ollama(
                model="llama3.2:3b", # Switched to the user-preferred small model
                base_url=self.config.ollama_host,
                temperature=0.2,
                request_timeout=30
            )

            response = enhancement_llm.complete(prompt)
            
            # Combine original question with enhanced versions
            enhanced_queries = response.text.strip().split('\\n')
            combined_query = question + " " + " ".join(enhanced_queries)
            
            return combined_query

        except Exception as e:
            self.logger.warning(f"LLM-based query enhancement failed: {e}. Using basic enhancement.")
            return question # Fallback to original question

    def _create_enhanced_prompt(self, question: str) -> str:
        """Create enhanced prompt for Dell SRM context"""
        prompt_template = """
You are a technical documentation expert for Dell SRM systems. Your primary goal is to provide accurate, factual answers based exclusively on the provided context.

ACCURACY REQUIREMENTS:
1.  **Answer ONLY based on the provided context.** Do not use any prior knowledge.
2.  If the context is insufficient to answer the question, you MUST explicitly state: "The provided documentation does not contain sufficient information to answer this question."
3.  Use the exact technical terms, component names, and procedures found in the documentation.
4.  When available, include specific steps, requirements, or configuration details in your answer.
5.  Cite source information when referencing specific details, if possible.

USER QUESTION:
{question}

Based on the user's question and the context that will be provided, formulate a comprehensive and technically accurate response. If the information is not available, state it clearly.

TECHNICAL RESPONSE:
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

        # Check if data is available
        data_available = result.get("data_available", True)

        # Display answer with appropriate styling
        if data_available:
            console.print("\n📋 Answer:")
            answer_panel = Panel(
                result["answer"],
                title="Dell SRM Assistant Response",
                border_style="green"
            )
        else:
            console.print("\n⚠️  Response:")
            answer_panel = Panel(
                result["answer"],
                title="No Data Available",
                border_style="yellow"
            )
        console.print(answer_panel)
        
        # Display sources if available and data is present
        if result.get("sources") and data_available:
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
        elif not data_available:
            console.print(f"\n📚 Sources: No relevant sources found in the documentation")
        
        # Display query time
        if result.get("query_time"):
            console.print(f"\n⏱️  Query time: {result['query_time']}s")
    
    def save_config(self, path: str):
        """Save current configuration"""
        config_dict = asdict(self.config)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        console.print(f"✅ Configuration saved to {path}")
    
    def benchmark_retrieval(self, test_queries: List[str] = None):
        """Benchmark retrieval performance with test queries"""
        if not test_queries:
            test_queries = [
                "What are the system requirements for Dell SRM?",
                "How do I configure SolutionPacks?",
                "What is the upgrade procedure?",
                "How do I troubleshoot host discovery?",
                "What monitoring capabilities are available?"
            ]
        
        console.print("\n🧪 Retrieval System Benchmark")
        console.print("=" * 40)
        
        results = []
        total_time = 0
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"\nTest {i}/{len(test_queries)}: {query}")
            
            start_time = time.time()
            result = self.query(query)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Extract retrieval stats
            retrieval_stats = {
                "query": query,
                "time": query_time,
                "sources_found": len(result.get("sources", [])),
                "data_available": result.get("data_available", False),
                "error": "error" in result
            }
            
            results.append(retrieval_stats)
            
            # Display quick result
            status = "✅" if retrieval_stats["data_available"] else "❌"
            console.print(f"   {status} {query_time:.2f}s - {retrieval_stats['sources_found']} sources")
        
        # Summary
        console.print(f"\n📊 Benchmark Summary:")
        console.print(f"   Total queries: {len(test_queries)}")
        console.print(f"   Average time: {total_time/len(test_queries):.2f}s")
        console.print(f"   Success rate: {sum(1 for r in results if r['data_available'])/len(results)*100:.1f}%")
        console.print(f"   Total sources found: {sum(r['sources_found'] for r in results)}")
        
        return results

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
        default="./documents",
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
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run retrieval system benchmark"
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
        
        # Benchmark mode
        if args.benchmark:
            rag.benchmark_retrieval()
            return 0
        
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
