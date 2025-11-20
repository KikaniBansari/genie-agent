# GenieSuite AI - Enterprise Business Strategy Assistant

<div align="center">

![GenieSuite AI](https://img.shields.io/badge/GenieSuite-AI-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/version-2.0.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge)

**Transform your business goals into actionable strategies with AI-powered intelligence**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

GenieSuite AI is an advanced, multi-agent business strategy assistant that leverages cutting-edge AI to help businesses transform their goals into comprehensive, actionable strategies. Built with a modern, Apple-inspired interface and powered by multiple LLM providers, GenieSuite delivers professional-grade strategic insights.

### Key Highlights

- 🤖 **Multi-Agent Architecture**: Research, Strategy, and Content agents work together
- 🎨 **Beautiful UI**: Apple-inspired design with dark/light mode
- 🔍 **Deep Research**: Integrated web search with multiple data sources
- 📊 **Comprehensive Reports**: Professional Word document generation
- 🚀 **Fast & Efficient**: Optimized for speed and reliability
- 🔒 **Secure**: Server-side API key management

---

## ✨ Features

### Frontend Features
- **Modern UI/UX**: Clean, minimal design inspired by Apple's aesthetic
- **Dark/Light Mode**: Seamless theme switching with system preference detection
- **Real-time Chat**: Interactive conversation interface with markdown support
- **Voice Input**: Speech-to-text for hands-free interaction
- **Export Options**: Download conversations and reports
- **Analytics Dashboard**: Track usage statistics and performance
- **Responsive Design**: Works beautifully on all devices
- **Toast Notifications**: User-friendly feedback system
- **Loading States**: Smooth animations and progress indicators

### Backend Features
- **Multi-LLM Support**: Groq, OpenAI, and Anthropic integration
- **Intelligent Research**: Advanced web search with Tavily API
- **Enhanced Error Handling**: Comprehensive error management and logging
- **Batch Processing**: Process multiple requests simultaneously
- **Feedback System**: Collect and analyze user feedback
- **Analytics API**: Track usage and performance metrics
- **Auto Cleanup**: Automatic cleanup of old files
- **Health Checks**: System monitoring endpoints

### Agent Capabilities
- **Research Agent**: Deep market research and trend analysis
- **Strategy Agent**: Comprehensive 5-step strategic planning
- **Content Agent**: LinkedIn posts and executive summaries
- **Action Items**: Extracted, actionable task lists
- **Source Attribution**: Credible source references

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js (for frontend development, optional)
- API Keys (at least one):
  - Groq API Key (recommended)
  - OpenAI API Key
  - Anthropic API Key
  - Tavily API Key (for web search)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/genie-suite.git
cd genie-suite
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Provider (at least one required)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Web Search
TAVILY_API_KEY=your_tavily_api_key_here
```

### Step 5: Run the Backend

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### Step 6: Open the Frontend

Simply open `frontend/index.html` in your web browser, or serve it using a local server:

```bash
# Using Python
cd frontend
python -m http.server 8080

# Using Node.js
npx http-server frontend -p 8080
```

Then navigate to `http://localhost:8080`

---

## 📖 Usage

### Basic Usage

1. **Start the Backend**: Ensure the FastAPI server is running
2. **Open Frontend**: Open the frontend in your browser
3. **Enter Your Goal**: Type your business goal and industry
4. **Get Strategy**: Receive a comprehensive strategy report
5. **Download Report**: Download the generated Word document

### Example Goals

- "Scale my coffee shop brand in Seattle"
- "Launch a SaaS product for small businesses"
- "Expand my consulting firm to new markets"
- "Improve customer retention for my e-commerce store"

### Quick Actions

The frontend includes quick action buttons for common business scenarios:
- Retail Business
- Tech Startup
- Consulting

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /
GET /api/health
```

#### Process Request
```http
POST /api/process
Content-Type: application/json

{
  "goal": "Scale my coffee shop brand in Seattle",
  "industry": "Retail"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "research_summary": "...",
    "strategy": "...",
    "content_sample": "...",
    "action_items": "...",
    "sources": [...]
  },
  "download_link": "/api/download/report_20240101_120000_abc123.docx",
  "request_id": "report_20240101_120000_abc123",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Batch Processing
```http
POST /api/process/batch
Content-Type: application/json

{
  "requests": [
    {"goal": "Goal 1", "industry": "Industry 1"},
    {"goal": "Goal 2", "industry": "Industry 2"}
  ]
}
```

#### Download Report
```http
GET /api/download/{filename}
```

#### List Reports
```http
GET /api/reports?limit=10
```

#### Submit Feedback
```http
POST /api/feedback
Content-Type: application/json

{
  "request_id": "report_20240101_120000_abc123",
  "rating": 5,
  "feedback": "Great strategy!"
}
```

#### Get Analytics
```http
GET /api/analytics
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

---

## 🏗️ Project Structure

```
genie-suite/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── logic.py         # GenieEngine and agent logic
│   └── __pycache__/
├── frontend/
│   ├── index.html       # Main HTML file
│   ├── style.css        # Styles (Apple-inspired design)
│   └── app.js           # Frontend JavaScript
├── outputs/             # Generated reports (created automatically)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🎨 Customization

### Changing LLM Provider

The system automatically detects available API keys. To prioritize a specific provider, modify the `_detect_llm_provider()` method in `backend/logic.py`.

### Customizing Prompts

Edit the prompt templates in `backend/logic.py`:
- `strategy_prompt`: Strategy generation prompt
- `content_prompt`: Content generation prompt
- `action_items_prompt`: Action items extraction prompt

### Styling

Modify `frontend/style.css` to customize the appearance. The design uses CSS variables for easy theming:

```css
:root {
    --accent: #0071e3;
    --bg-primary: #ffffff;
    /* ... */
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | No* | Groq API key for LLM |
| `OPENAI_API_KEY` | No* | OpenAI API key for LLM |
| `ANTHROPIC_API_KEY` | No* | Anthropic API key for LLM |
| `TAVILY_API_KEY` | No | Tavily API key for web search |

*At least one LLM API key is required.

### API Keys Setup

1. **Groq**: Sign up at https://console.groq.com
2. **OpenAI**: Get key from https://platform.openai.com
3. **Anthropic**: Get key from https://console.anthropic.com
4. **Tavily**: Sign up at https://tavily.com

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: API returns "Error: API Key missing"
- **Solution**: Ensure your `.env` file contains at least one LLM API key

**Problem**: Search results are empty
- **Solution**: Check your Tavily API key, or the system will use fallback mode

**Problem**: Document generation fails
- **Solution**: Ensure the `outputs/` directory is writable

### Frontend Issues

**Problem**: Cannot connect to backend
- **Solution**: Update `API_URL` in `frontend/app.js` to match your backend URL

**Problem**: Voice input not working
- **Solution**: Ensure you're using a browser that supports Web Speech API (Chrome, Edge)

**Problem**: Theme not persisting
- **Solution**: Check browser localStorage permissions

---

## 🚀 Deployment

### Backend Deployment

#### Using Render
1. Connect your GitHub repository
2. Set environment variables in Render dashboard
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `cd backend && python main.py`

#### Using Heroku
```bash
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
heroku config:set TAVILY_API_KEY=your_key
git push heroku main
```

### Frontend Deployment

#### Using GitHub Pages
1. Push frontend files to `gh-pages` branch
2. Update `API_URL` in `app.js` to your backend URL
3. Enable GitHub Pages in repository settings

#### Using Netlify/Vercel
1. Connect your repository
2. Set build directory to `frontend`
3. Update `API_URL` in `app.js`

---

## 📊 Performance

- **Average Response Time**: 2-5 seconds
- **Concurrent Requests**: Supports multiple simultaneous requests
- **File Cleanup**: Automatic cleanup of files older than 24 hours
- **Caching**: Research results can be cached (future enhancement)

---

## 🔒 Security

- API keys are stored server-side only
- Input validation on all endpoints
- CORS configuration for production
- File path sanitization
- Rate limiting (recommended for production)

### Production Checklist

- [ ] Set specific CORS origins
- [ ] Enable rate limiting
- [ ] Use HTTPS
- [ ] Secure API keys
- [ ] Enable logging
- [ ] Set up monitoring

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
cd backend
uvicorn main:app --reload

# Run tests (when available)
pytest
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **Tavily** for web search capabilities
- **FastAPI** for the excellent web framework
- **Apple** for design inspiration

---

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] PDF export option
- [ ] Email report delivery
- [ ] Integration with CRM systems
- [ ] Mobile app
- [ ] Team collaboration features
- [ ] Custom agent training

---

<div align="center">

**Made with ❤️ using AI**

[⭐ Star this repo](https://github.com/yourusername/genie-suite) • [🐛 Report Bug](https://github.com/yourusername/genie-suite/issues) • [💡 Request Feature](https://github.com/yourusername/genie-suite/issues)

</div>

