# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The goal of this app is to read an excel file with data and produce interactive analytics
The front end is a web app that can accept files in Excel format
The back end file will be a Python file that will analyze the data and draw charts and produce relevant analytics

## Architecture

- **Frontend**: Web application for file upload and displaying analytics
- **Backend**: Python service for data processing and chart generation
- **Data Flow**: Excel file → Upload → Python analysis → Interactive visualizations

## Development Setup

### Python Backend (Implemented)

- Dependencies managed via `requirements.txt`
- Current stack: Flask, Pandas, Plotly, scikit-learn, OpenPyXL
- Uses virtual environments: `python -m venv venv`

### Frontend (Implemented)

- Pure HTML/CSS/JavaScript with Plotly.js
- File upload capabilities for Excel files (.xlsx/.xls)
- Interactive visualizations with Plotly charts

### Production Deployment (Railway)

- Application deployed at: `https://web-production-00170.up.railway.app`
- Automatic deployment from GitHub
- Configuration files: `Procfile`, `runtime.txt`

## Common Commands

### Local Development

```bash
pip install -r requirements.txt  # Install dependencies
python app.py                    # Run Flask server (http://localhost:5000)
```

### Production Deployment (Railway)

```bash
git add .                        # Stage changes
git commit -m "Your message"     # Commit changes
git push origin main             # Push to GitHub (triggers Railway deploy)
```

### Railway Configuration Files

- `Procfile`: `web: python app.py`
- `runtime.txt`: `python-3.11`
- `app.py`: Modified to use `PORT` environment variable

## Key Implementation Notes

- Excel file processing should handle various formats (.xlsx, .xls)
- Consider file size limits and streaming for large datasets
- Implement error handling for malformed Excel files
- Design API endpoints for data analysis requests
- Plan for scalable chart generation (server-side vs client-side rendering)
