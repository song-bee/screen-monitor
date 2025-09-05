#!/usr/bin/env python3
"""
Development Environment Setup Script

This script sets up the ASAM development environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, cwd=None):
    """Run shell command and return result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, cwd=cwd, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result


def setup_python_environment():
    """Set up Python virtual environment and dependencies"""
    print("Setting up Python environment...")

    # Check if venv exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command("python3 -m venv venv")

    # Install dependencies
    print("Installing Python dependencies...")
    run_command("source venv/bin/activate && pip install --upgrade pip")
    run_command("source venv/bin/activate && pip install -e .[dev]")

    # Install platform-specific dependencies
    if sys.platform == "darwin":
        print("Installing macOS-specific dependencies...")
        run_command("source venv/bin/activate && pip install -e .[macos]")

    print("✅ Python environment setup complete")


def setup_pre_commit():
    """Set up pre-commit hooks"""
    print("Setting up pre-commit hooks...")

    run_command("source venv/bin/activate && pre-commit install")
    run_command("source venv/bin/activate && pre-commit install --hook-type commit-msg")

    print("✅ Pre-commit hooks installed")


def check_system_dependencies():
    """Check for required system dependencies"""
    print("Checking system dependencies...")

    # Check for Ollama
    try:
        result = run_command("ollama --version", check=False)
        if result.returncode != 0:
            print(
                "⚠️  Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh"
            )
        else:
            print("✅ Ollama found")
    except:
        print(
            "⚠️  Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh"
        )

    # Check for required system tools on macOS
    if sys.platform == "darwin":
        try:
            run_command("which terminal-notifier", check=False)
            print("✅ terminal-notifier found")
        except:
            print(
                "⚠️  terminal-notifier not found. Install with: brew install terminal-notifier"
            )


def create_config_directories():
    """Create necessary configuration directories"""
    print("Creating configuration directories...")

    config_dirs = [
        Path.home() / ".asam",
        Path.home() / ".asam" / "logs",
        Path.home() / ".asam" / "models",
        Path.home() / ".asam" / "cache",
    ]

    for dir_path in config_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

    print("✅ Configuration directories created")


def main():
    """Main setup function"""
    print("🚀 Setting up ASAM development environment...")
    print("=" * 50)

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    try:
        setup_python_environment()
        setup_pre_commit()
        check_system_dependencies()
        create_config_directories()

        print("\n" + "=" * 50)
        print("🎉 Development environment setup complete!")
        print("\nNext steps:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Install Ollama LLM model: ollama pull llama3.2:3b")
        print("3. Run tests: python -m pytest tests/")
        print("4. Start development: python -m asam.main --dev-mode")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
