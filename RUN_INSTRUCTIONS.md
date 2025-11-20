# How to Run GenieSuite on Windows CMD

## Prerequisites
- Python 3.8 or higher installed
- Gemini API Key (Get it from: https://makersuite.google.com/app/apikey)
- Tavily API Key (Optional, for web search: https://tavily.com)

## Step-by-Step Instructions

### Step 1: Open Command Prompt
1. Press `Win + R`
2. Type `cmd` and press Enter
3. Navigate to your project directory:
   ```cmd
   cd C:\Users\bansa\OneDrive\Desktop\genie-suite
   ```

### Step 2: Activate Virtual Environment
If you have a virtual environment (venv folder exists):
```cmd
venv\Scripts\activate
```

If you don't have a virtual environment, create one:
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Step 3: Create .env File
Create a `.env` file in the root directory with your API keys:

```cmd
echo GEMINI_API_KEY=your_gemini_api_key_here > .env
echo TAVILY_API_KEY=your_tavily_api_key_here >> .env
```

Or manually create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Note:** Replace `your_gemini_api_key_here` with your actual Gemini API key. Tavily API key is optional.

### Step 4: Install Dependencies
Install all required packages:
```cmd
pip install -r requirements.txt
```

This will install:
- FastAPI
- Uvicorn
- LangChain
- LangChain Google GenAI
- Other dependencies

### Step 5: Run the Backend Server
Navigate to the backend directory and start the server:
```cmd
cd backend
python main.py
```

Or run from the root directory:
```cmd
python backend\main.py
```

You should see output like:
```
INFO:     Started server process [xxxxx]
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 6: Open the Frontend
1. Open your web browser
2. Navigate to: `http://localhost:8000`
   Or open the frontend HTML file directly:
   ```
   frontend\index.html
   ```
   (Right-click on `frontend\index.html` and select "Open with" → your browser)

### Step 7: Test the Application
1. Open the frontend in your browser
2. Enter a business goal (e.g., "Increase online sales by 50%")
3. Select an industry
4. Click "Generate Strategy"
5. Wait for the AI to generate your strategy report

## Troubleshooting

### Issue: Module not found errors
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Port 8000 already in use
**Solution:** Either stop the other application using port 8000, or change the port in `backend\main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
```

### Issue: API Key errors
**Solution:** Check that your `.env` file exists in the root directory and contains valid API keys:
```cmd
type .env
```

### Issue: Gemini API errors
**Solution:** 
- Verify your Gemini API key is correct
- Check your API quota/limits
- Ensure you have internet connectivity

## Quick Start Commands (All in one)

```cmd
cd C:\Users\bansa\OneDrive\Desktop\genie-suite
venv\Scripts\activate
pip install -r requirements.txt
cd backend
python main.py
```

Then open `frontend\index.html` in your browser.

## Stopping the Server
Press `Ctrl + C` in the command prompt window where the server is running.

