import os
import json
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
from docx import Document

load_dotenv()

# --- Configuration & Models ---
@dataclass
class ResearchResult:
    summary: str
    sources: List[Dict]

class GenieEngine:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        
    def _llm_generate(self, prompt: str) -> str:
        """Secure Server-Side LLM Call"""
        if not self.groq_key: return "Error: API Key missing."
        
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def search_web(self, query: str) -> List[Dict]:
        """Secure Server-Side Search"""
        if not self.tavily_key: return []
        try:
            resp = requests.post("https://api.tavily.com/search", 
                               json={"api_key": self.tavily_key, "query": query, "max_results": 3})
            return resp.json().get("results", [])
        except:
            return []

    def run_pipeline(self, goal: str, industry: str) -> Dict[str, Any]:
        """The Orchestrator"""
        
        # 1. Research Phase
        search_results = self.search_web(f"{industry} trends {goal}")
        context_text = "\n".join([r['content'] for r in search_results])
        
        # 2. Strategy Phase
        strategy_prompt = f"""
        Context: {context_text[:2000]}
        Goal: {goal}
        Industry: {industry}
        
        Generate a concise 3-step executive strategy.
        """
        strategy = self._llm_generate(strategy_prompt)
        
        # 3. Content Phase (Simplified for demo)
        content_prompt = f"Based on this strategy: {strategy}, write one LinkedIn post."
        post = self._llm_generate(content_prompt)
        
        return {
            "research_summary": context_text[:500] + "...",
            "strategy": strategy,
            "content_sample": post,
            "sources": search_results
        }

    def generate_doc(self, data: Dict, filename: str):
        """Generates physical document"""
        doc = Document()
        doc.add_heading('GenieSuite Report', 0)
        doc.add_paragraph(f"Generated: {datetime.now()}")
        
        doc.add_heading('Strategy', level=1)
        doc.add_paragraph(data['strategy'])
        
        doc.add_heading('Content Draft', level=1)
        doc.add_paragraph(data['content_sample'])
        
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", filename)
        doc.save(path)
        return path