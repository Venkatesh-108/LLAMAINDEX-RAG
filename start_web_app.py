#!/usr/bin/env python3
"""
Startup script for SRM AI Doc Assist Web Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_socketio
        print("✅ Web dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing web dependencies: {e}")
        print("Please install web dependencies first:")
        print("pip install flask flask-socketio python-socketio python-engineio")
        return False

def check_ollama():
    """Check if Ollama is running"""
    try:
        import ollama
        client = ollama.Client()
        client.list()
        print("✅ Ollama is running")
        return True
    except Exception as e:
        print("⚠️ Ollama is not running or not accessible")
        print("The web app will run in demo mode")
        return False

def main():
    """Main startup function"""
    print("🚀 SRM AI Doc Assist - Web Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check Ollama
    check_ollama()
    
    # Check if web app file exists
    web_app_path = Path("web_app.py")
    if not web_app_path.exists():
        print("❌ web_app.py not found")
        print("Please ensure you're in the correct directory")
        return 1
    
    # Check if templates directory exists
    templates_path = Path("templates")
    if not templates_path.exists():
        print("❌ templates directory not found")
        print("Please ensure you're in the correct directory")
        return 1
    
    print("\n📱 Starting web application...")
    print("🌐 The application will be available at: http://localhost:5000")
    print("📖 Make sure your PDF documents are in the 'documents' folder")
    print("\nPress Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Start the web application
        subprocess.run([sys.executable, "web_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Web application stopped")
        return 0
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
