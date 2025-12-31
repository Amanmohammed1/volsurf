#!/usr/bin/env python3
"""
Run script for Heston Calibration Application
Starts FastAPI backend and serves frontend
"""

import subprocess
import webbrowser
import time
import sys
import os

def main():
    print("=" * 60)
    print("  HESTON MODEL CALIBRATION - QUANT PORTFOLIO")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('backend/main.py'):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check for token
    if not os.environ.get('UPSTOX_ACCESS_TOKEN'):
        # Try loading from .env
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            print("[0/3] Loading environment from .env...")
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
    
    print("[1/3] Starting FastAPI backend server...")
    
    # Start FastAPI backend with uvicorn
    backend_process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '5001'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    time.sleep(3)  # Wait for server to start
    
    print("[2/3] Starting frontend server...")
    
    # Start simple HTTP server for frontend
    frontend_process = subprocess.Popen(
        [sys.executable, '-m', 'http.server', '8080'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    time.sleep(1)
    
    print("[3/3] Opening browser...")
    print()
    print("-" * 60)
    print("  Backend API:  http://localhost:5001")
    print("  API Docs:     http://localhost:5001/docs")
    print("  Frontend:     http://localhost:8080")
    print("-" * 60)
    print()
    print("Press Ctrl+C to stop both servers")
    print()
    
    # Open browser
    webbrowser.open('http://localhost:8080')
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done!")


if __name__ == '__main__':
    main()
