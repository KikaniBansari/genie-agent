const API_URL = "http://localhost:8000/api"; // Change this to your Render/Cloud URL for production

// DOM Elements
const chatHistory = document.getElementById('chatHistory');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const visualizer = document.getElementById('visualizer');

// Voice Setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const synthesis = window.speechSynthesis;
let recognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        voiceBtn.classList.add('recording');
    };
    
    recognition.onend = () => {
        voiceBtn.classList.remove('recording');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        handleInput(); // Auto-send on voice end
    };
} else {
    voiceBtn.style.display = 'none'; // Hide if not supported
}

// Helpers
function addMessage(text, sender, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    
    const avatar = sender === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
    
    // Use Marked.js for bot responses to render formatting
    const contentBody = isHtml ? text : (sender === 'bot' ? marked.parse(text) : text);
    
    div.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="content">${contentBody}</div>
    `;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function speak(text) {
    if (synthesis.speaking) synthesis.cancel();
    
    // Strip HTML tags for speaking
    const cleanText = text.replace(/<[^>]*>?/gm, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    
    utterance.onstart = () => visualizer.classList.add('active');
    utterance.onend = () => visualizer.classList.remove('active');
    
    synthesis.speak(utterance);
}

async function handleInput() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    userInput.value = '';
    
    // Show loading state
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `<div class="avatar"><i class="fa-solid fa-robot"></i></div><div class="content"><i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing market data...</div>`;
    chatHistory.appendChild(loadingDiv);

    try {
        // Call Backend
        const response = await fetch(`${API_URL}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: text,
                industry: "General Business" // In a full app, extract this using NLP
            })
        });

        const data = await response.json();
        document.getElementById(loadingId).remove();

        if (data.status === 'success') {
            const botReply = `**Strategy Generated:**\n\n${data.data.strategy}\n\n[<i class="fa-solid fa-file-arrow-down"></i> Download Full Report](${API_URL.replace('/api','')}${data.download_link})`;
            
            addMessage(botReply, 'bot');
            speak(data.data.strategy.substring(0, 200) + "..."); // Speak summary
        } else {
            addMessage("I encountered an error processing your request.", 'bot');
        }

    } catch (err) {
        document.getElementById(loadingId).remove();
        addMessage(`Connection Error: Ensure the backend server is running.`, 'bot');
    }
}

// Event Listeners
sendBtn.addEventListener('click', handleInput);
userInput.addEventListener('keypress', (e) => { if(e.key === 'Enter') handleInput(); });
voiceBtn.addEventListener('click', () => {
    if (recognition) recognition.start();
});