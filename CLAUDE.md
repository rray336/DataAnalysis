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

This project appears to be in early development. When setting up:

### Python Backend

- Use `requirements.txt` or `pyproject.toml` for dependency management
- Common dependencies will likely include: pandas, openpyxl, matplotlib, plotly, fastapi/flask
- Consider using virtual environments: `python -m venv venv`

### Frontend Setup

- Choose framework (React, Vue, or vanilla HTML/JS)
- Include file upload capabilities for Excel files
- Set up visualization libraries (Chart.js, D3.js, or integrate with Python plots)

## Common Commands

When project structure is established, typical commands will be:

### Python

```bash
pip install -r requirements.txt  # Install dependencies
python app.py                    # Run backend server
python -m pytest                 # Run tests (when implemented)
```

### Frontend (depending on framework choice)

```bash
npm install                      # Install dependencies
npm run dev                      # Development server
npm run build                    # Production build
npm test                         # Run tests
```

## Key Implementation Notes

- Excel file processing should handle various formats (.xlsx, .xls)
- Consider file size limits and streaming for large datasets
- Implement error handling for malformed Excel files
- Design API endpoints for data analysis requests
- Plan for scalable chart generation (server-side vs client-side rendering)
