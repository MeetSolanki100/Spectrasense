#!/usr/bin/env python3
"""
Development Helper Script for Voice Assistant
Provides quick commands for common development tasks
"""

import sys
import subprocess
import os
import platform

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_requirements():
    """Check if all requirements are met"""
    print_header("Checking Requirements")
    
    checks = {
        "Python": ["python", "--version"],
        "Node.js": ["node", "--version"],
        "npm": ["npm", "--version"],
        "Ollama": ["ollama", "--version"]
    }
    
    results = {}
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ {name}: {version}")
                results[name] = True
            else:
                print(f"❌ {name}: Not found")
                results[name] = False
        except FileNotFoundError:
            print(f"❌ {name}: Not installed")
            results[name] = False
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama: Running")
            results["Ollama Running"] = True
        else:
            print("⚠️  Ollama: Not running (start with 'ollama serve')")
            results["Ollama Running"] = False
    except:
        print("⚠️  Ollama: Not running (start with 'ollama serve')")
        results["Ollama Running"] = False
    
    return all(results.values())

def setup_backend():
    """Setup backend environment"""
    print_header("Setting Up Backend")
    
    os.chdir("backend")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    
    # Activate and install dependencies
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    print("Installing dependencies...")
    subprocess.run([pip_path, "install", "-r", "../requirements.txt"])
    print("✅ Backend setup complete")
    
    os.chdir("..")

def setup_frontend():
    """Setup frontend environment"""
    print_header("Setting Up Frontend")
    
    os.chdir("frontend")
    
    if not os.path.exists("node_modules"):
        print("Installing npm dependencies...")
        subprocess.run(["npm", "install"])
        subprocess.run(["npm", "install", "lucide-react"])
        print("✅ Frontend setup complete")
    else:
        print("✅ Dependencies already installed")
    
    os.chdir("..")

def start_backend():
    """Start the backend server"""
    print_header("Starting Backend Server")
    print("Backend will run on http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    os.chdir("backend")
    
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    try:
        subprocess.run([python_path, "api.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Backend stopped")
    
    os.chdir("..")

def start_frontend():
    """Start the frontend dev server"""
    print_header("Starting Frontend Dev Server")
    print("Frontend will run on http://localhost:5173")
    print("Press Ctrl+C to stop\n")
    
    os.chdir("frontend")
    
    try:
        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n\n✅ Frontend stopped")
    
    os.chdir("..")

def run_tests():
    """Run API tests"""
    print_header("Running API Tests")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("❌ Backend is not responding. Please start it first.")
            return
    except:
        print("❌ Backend is not running. Please start it first.")
        print("Run: python dev.py start-backend")
        return
    
    subprocess.run([sys.executable, "test_api.py"])

def clean_project():
    """Clean build artifacts and caches"""
    print_header("Cleaning Project")
    
    items_to_clean = [
        ("backend/__pycache__", "Backend cache"),
        ("components/__pycache__", "Components cache"),
        ("frontend/dist", "Frontend build"),
        ("frontend/node_modules", "Node modules (will need reinstall)"),
    ]
    
    for path, description in items_to_clean:
        if os.path.exists(path):
            response = input(f"Delete {description} ({path})? [y/N]: ")
            if response.lower() == 'y':
                import shutil
                shutil.rmtree(path)
                print(f"✅ Deleted {path}")
        else:
            print(f"⏭️  {path} doesn't exist")

def show_status():
    """Show project status"""
    print_header("Project Status")
    
    # Backend status
    backend_venv = os.path.exists("backend/venv")
    print(f"Backend virtual environment: {'✅ Present' if backend_venv else '❌ Missing'}")
    
    # Frontend status
    frontend_deps = os.path.exists("frontend/node_modules")
    print(f"Frontend dependencies: {'✅ Installed' if frontend_deps else '❌ Not installed'}")
    
    # Database status
    db_exists = os.path.exists("chroma_db")
    print(f"ChromaDB: {'✅ Initialized' if db_exists else '⚠️  Not initialized'}")
    
    # Check if services are running
    try:
        import requests
        backend_response = requests.get("http://localhost:8000/health", timeout=1)
        print(f"Backend API: {'✅ Running' if backend_response.status_code == 200 else '❌ Not running'}")
    except:
        print("Backend API: ❌ Not running")
    
    try:
        import requests
        frontend_response = requests.get("http://localhost:5173", timeout=1)
        print(f"Frontend: {'✅ Running' if frontend_response.status_code == 200 else '❌ Not running'}")
    except:
        print("Frontend: ❌ Not running")
    
    try:
        import requests
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=1)
        print(f"Ollama: {'✅ Running' if ollama_response.status_code == 200 else '❌ Not running'}")
    except:
        print("Ollama: ❌ Not running")

def show_help():
    """Show available commands"""
    print_header("Voice Assistant - Development Helper")
    
    commands = {
        "check": "Check if all requirements are installed",
        "setup": "Setup both backend and frontend",
        "setup-backend": "Setup only backend",
        "setup-frontend": "Setup only frontend",
        "start-backend": "Start the backend server",
        "start-frontend": "Start the frontend dev server",
        "test": "Run API tests",
        "clean": "Clean build artifacts and caches",
        "status": "Show project status",
        "help": "Show this help message"
    }
    
    print("Usage: python dev.py [command]\n")
    print("Available commands:\n")
    
    for cmd, description in commands.items():
        print(f"  {cmd:20} - {description}")
    
    print("\nExample workflow:")
    print("  1. python dev.py check")
    print("  2. python dev.py setup")
    print("  3. python dev.py start-backend  (in terminal 1)")
    print("  4. python dev.py start-frontend (in terminal 2)")
    print("  5. Open http://localhost:5173 in browser")

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        "check": check_requirements,
        "setup": lambda: (setup_backend(), setup_frontend()),
        "setup-backend": setup_backend,
        "setup-frontend": setup_frontend,
        "start-backend": start_backend,
        "start-frontend": start_frontend,
        "test": run_tests,
        "clean": clean_project,
        "status": show_status,
        "help": show_help
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python dev.py help' to see available commands")

if __name__ == "__main__":
    main()
