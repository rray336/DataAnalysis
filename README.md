# Excel Data Analytics App

A web application that allows users to upload Excel files and generate interactive analytics and visualizations.

## Features

- **File Upload**: Support for .xlsx and .xls files
- **Data Analysis**: Automatic generation of summary statistics
- **Interactive Visualizations**: 
  - Data preview table
  - Distribution histograms for numeric columns
  - Correlation matrix
  - Missing data analysis
- **Responsive Design**: Clean, modern web interface

## Setup and Installation

### Local Development

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Production Deployment (Railway)

This application is configured for deployment on Railway:

1. **Push to GitHub repository**
2. **Deploy on Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway automatically detects and deploys the Flask app

3. **Live Application**: 
   ```
   https://web-production-00170.up.railway.app
   ```

#### Railway Configuration Files:
- `Procfile` - Defines the web process
- `runtime.txt` - Specifies Python 3.11
- Updated `app.py` - Uses Railway's PORT environment variable

## Usage

1. Click "Choose Excel File" to select your .xlsx or .xls file
2. Click "Upload & Analyze" to process the file
3. View the generated analytics and interactive charts

## File Size Limits

- Maximum file size: 16MB
- Supported formats: .xlsx, .xls

## Dependencies

- Flask: Web framework
- Pandas: Data manipulation and analysis
- Plotly: Interactive visualizations
- OpenPyXL: Excel file reading
- Flask-CORS: Cross-origin resource sharing

## Project Structure

```
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── Procfile           # Railway deployment configuration
├── runtime.txt        # Python version specification
├── templates/
│   └── index.html     # Frontend interface
└── uploads/           # Temporary file storage (created automatically)
```