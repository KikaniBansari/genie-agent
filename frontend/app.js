const API_URL = "http://127.0.0.1:8000/api";

// State Management
const state = {
    voiceEnabled: localStorage.getItem('voiceEnabled') !== 'false',
    autoScroll: true,
    isProcessing: false
};

// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const attachBtn = document.getElementById('attachBtn');
const welcomeScreen = document.getElementById('welcomeScreen');
const quickOptions = document.getElementById('quickOptions');
const loadingOverlay = document.getElementById('loadingOverlay');
const toastContainer = document.getElementById('toastContainer');

// Voice Setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const synthesis = window.speechSynthesis;
let recognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        voiceBtn.classList.add('recording');
        showToast('Listening...', 'info');
    };
    
    recognition.onend = () => {
        voiceBtn.classList.remove('recording');
    };
    
    recognition.onerror = (event) => {
        voiceBtn.classList.remove('recording');
        showToast('Voice recognition error. Please try again.', 'error');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        handleInput();
    };
} else {
    voiceBtn.style.display = 'none';
}

// Initialize App
function init() {
    setupEventListeners();
    userInput.focus();
}

// Event Listeners Setup
function setupEventListeners() {
    // Send button
    sendBtn.addEventListener('click', handleInput);
    
    // Enter key
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleInput();
        }
    });

    // Voice input
    voiceBtn.addEventListener('click', () => {
        if (recognition && !state.isProcessing) {
            recognition.start();
        }
    });

    // Quick option chips
    document.querySelectorAll('.option-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const example = chip.dataset.example;
            userInput.value = example;
            userInput.focus();
            handleInput();
        });
    });

    // Attach button (placeholder for future file upload)
    attachBtn.addEventListener('click', () => {
        showToast('File attachment coming soon!', 'info');
    });

    // Auto-hide quick options when user starts typing
    userInput.addEventListener('input', () => {
        if (userInput.value.trim().length > 0) {
            quickOptions.classList.add('hidden');
        } else {
            quickOptions.classList.remove('hidden');
        }
    });
}

// Message Handling
function addMessage(text, sender, isHtml = false) {
    // Hide welcome screen
    if (welcomeScreen && !welcomeScreen.classList.contains('hidden')) {
        welcomeScreen.classList.add('hidden');
    }

    // Hide quick options when message is sent
    if (sender === 'user') {
        quickOptions.classList.add('hidden');
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatarIcon = sender === 'bot' 
        ? '<i class="fa-solid fa-sparkles"></i>' 
        : '<i class="fa-solid fa-user"></i>';
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let contentBody;
    if (isHtml) {
        contentBody = text;
    } else if (sender === 'bot') {
        contentBody = marked.parse(text);
    } else {
        contentBody = escapeHtml(text);
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarIcon}</div>
        <div class="message-content">
            ${contentBody}
            <div class="message-timestamp">${timestamp}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    if (state.autoScroll) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Show quick options again after bot response
    if (sender === 'bot') {
        setTimeout(() => {
            quickOptions.classList.remove('hidden');
        }, 500);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="message-avatar"><i class="fa-solid fa-sparkles"></i></div>
        <div class="message-content">
            <div class="message-loading">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    if (state.autoScroll) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function removeLoadingMessage() {
    const loading = document.getElementById('loading-message');
    if (loading) {
        loading.remove();
    }
}

// Input Handling
async function handleInput() {
    const text = userInput.value.trim();
    if (!text || state.isProcessing) return;

    state.isProcessing = true;
    sendBtn.disabled = true;
    
    addMessage(text, 'user');
    userInput.value = '';
    userInput.blur();
    
    showLoadingMessage();
    showLoadingOverlay(true);

    try {
        const response = await fetch(`${API_URL}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: text,
                industry: extractIndustry(text)
            })
        });

        const data = await response.json();
        removeLoadingMessage();
        showLoadingOverlay(false);

        if (data.status === 'success') {
            let botReply = `**Strategy Generated**\n\n${data.data.strategy}\n\n`;
            
            if (data.data.research_summary) {
                botReply += `**Research Summary:**\n${data.data.research_summary}\n\n`;
            }
            
            if (data.data.action_items) {
                botReply += `**Action Items:**\n${data.data.action_items}\n\n`;
            }
            
            if (data.data.content_sample) {
                botReply += `**Content Sample:**\n${data.data.content_sample}\n\n`;
            }
            
            if (data.download_link) {
                botReply += `[📥 Download Full Report](${API_URL.replace('/api', '')}${data.download_link})`;
            }
            
            addMessage(botReply, 'bot');
            
            if (state.voiceEnabled) {
                speak(data.data.strategy.substring(0, 200) + "...");
            }

            showToast('Strategy generated successfully!', 'success');
        } else {
            addMessage("I encountered an error processing your request. Please try again.", 'bot');
            showToast('Error processing request', 'error');
        }

    } catch (err) {
        removeLoadingMessage();
        showLoadingOverlay(false);
        addMessage(`Connection Error: ${err.message}. Please ensure the backend server is running at ${API_URL.replace('/api', '')}`, 'bot');
        showToast('Connection error. Check backend server.', 'error');
        console.error('Error:', err);
    } finally {
        state.isProcessing = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// Extract industry from text (simple NLP)
function extractIndustry(text) {
    const industries = {
        'tech': ['software', 'saas', 'tech', 'technology', 'app', 'digital', 'startup'],
        'retail': ['retail', 'shop', 'store', 'coffee', 'restaurant', 'food', 'cafe'],
        'consulting': ['consulting', 'consultant', 'advisory', 'strategy'],
        'healthcare': ['health', 'medical', 'hospital', 'clinic'],
        'finance': ['finance', 'banking', 'investment', 'financial'],
        'education': ['education', 'school', 'university', 'learning'],
        'e-commerce': ['e-commerce', 'ecommerce', 'online store', 'online shop'],
        'marketing': ['marketing', 'advertising', 'promotion', 'brand']
    };

    const lowerText = text.toLowerCase();
    for (const [industry, keywords] of Object.entries(industries)) {
        if (keywords.some(keyword => lowerText.includes(keyword))) {
            return industry.charAt(0).toUpperCase() + industry.slice(1).replace('-', ' ');
        }
    }
    return 'General Business';
}

// Voice Output
function speak(text) {
    if (!state.voiceEnabled || synthesis.speaking) return null;
    
    const cleanText = text.replace(/<[^>]*>?/gm, '').replace(/\*\*/g, '').replace(/\[.*?\]\(.*?\)/g, ''); // Remove markdown links
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 0.8;
    
    utterance.onstart = () => {
        voiceBtn.classList.add('speaking');
        // Visual feedback can be added here
    };
    utterance.onend = () => {
        voiceBtn.classList.remove('speaking');
        // Visual feedback can be removed here
    };
    utterance.onerror = () => {
        voiceBtn.classList.remove('speaking');
        // Handle error
    };
    
    synthesis.speak(utterance);
    return utterance; // Return utterance so caller can attach events
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info'
    };
    
    toast.innerHTML = `
        <i class="fa-solid ${icons[type] || icons.info}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Loading Overlay
function showLoadingOverlay(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

// Handle page visibility
document.addEventListener('visibilitychange', () => {
    if (document.hidden && synthesis.speaking) {
        synthesis.cancel();
    }
});

// Auto-focus input when clicking anywhere
document.addEventListener('click', (e) => {
    if (!e.target.closest('.message-content') && !e.target.closest('.input-wrapper')) {
        userInput.focus();
    }
});
