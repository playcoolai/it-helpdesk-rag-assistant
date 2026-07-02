"""
Environment check script — run this first to see what's already installed
and what's missing before we build Project 1 (IT Helpdesk RAG Assistant).

Usage:
    python check_environment.py
"""

import subprocess
import sys
import shutil

def check_command(name, command):
    path = shutil.which(command)
    if path:
        try:
            result = subprocess.run(
                [command, "--version"],
                capture_output=True, text=True, timeout=10
            )
            version = (result.stdout or result.stderr).strip().splitlines()[0]
            print(f"[OK]   {name:<15} -> {version}")
        except Exception:
            print(f"[OK]   {name:<15} -> found at {path} (version check failed)")
    else:
        print(f"[MISS] {name:<15} -> not found on PATH")

def check_python_package(package_name):
    try:
        __import__(package_name.replace("-", "_"))
        print(f"[OK]   python package '{package_name}' is installed")
    except ImportError:
        print(f"[MISS] python package '{package_name}' is NOT installed")

def check_ollama_models():
    path = shutil.which("ollama")
    if not path:
        print("[MISS] Ollama not found, skipping model list")
        return
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        print("\n--- Ollama models currently pulled ---")
        print(result.stdout.strip() or "(none found — you'll need to pull a model)")
    except Exception as e:
        print(f"[MISS] Could not run 'ollama list': {e}")

print("=" * 60)
print("SYSTEM TOOLS")
print("=" * 60)
check_command("Python", "python")
check_command("Pip", "pip")
check_command("Git", "git")
check_command("VSCode CLI", "code")
check_command("Ollama", "ollama")

print()
print("=" * 60)
print("PYTHON PACKAGES (for LangChain + RAG + Streamlit)")
print("=" * 60)
for pkg in ["langchain", "langchain_community", "langchain_ollama", "chromadb", "streamlit"]:
    check_python_package(pkg)

print()
print("=" * 60)
print("OLLAMA MODELS")
print("=" * 60)
check_ollama_models()

print()
print("=" * 60)
print(f"Python executable in use: {sys.executable}")
print("=" * 60)