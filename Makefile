.PHONY: help run start open test clean setup

# Default target when running 'make'
all: run

help:
	@echo "Fake NEWS Detection Using AI - Fake News Detector"
	@echo "Available commands:"
	@echo "  make run       - Start the Flask ML backend API server"
	@echo "  make open      - Open the frontend web app in your browser"
	@echo "  make test      - Run automated API and ML engine tests"
	@echo "  make clean     - Remove Python cache files"

run: start

start:
	@echo "Starting Fake NEWS Detection Using AI Backend Server..."
	./venv/bin/python3 backend/app.py

open:
	@echo "Opening Frontend Web App..."
	open frontend/index.html

test:
	@echo "Testing ML Engine & Flask API..."
	./venv/bin/python3 -c "import sys; sys.path.append('backend'); from app import app; client = app.test_client(); res = client.post('/api/predict', json={'text': 'Miracle cure eradicated cancer overnight!'}); print('API Test Output:', res.get_json())"

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
