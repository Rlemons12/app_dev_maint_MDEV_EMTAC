// ===============================
// Chatbot Frontend Script
// ===============================

// Global debounce timeout for submissions
let submissionTimeout = null;

// Text-to-speech state
let isTextToSpeechEnabled = false;
let voiceSelect;

// ===============================
// Voice Setup
// ===============================
function populateVoiceList() {
    if (!('speechSynthesis' in window)) return;
    if (!voiceSelect) {
        voiceSelect = document.getElementById('voice-select');
        if (!voiceSelect) {
            console.warn("No #voice-select element found, skipping voice list population.");
            return;
        }
    }

    const voices = window.speechSynthesis.getVoices();
    voiceSelect.innerHTML = "";

    voices.forEach((voice, i) => {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `${voice.name} (${voice.lang})`;
        voiceSelect.appendChild(option);
    });
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = populateVoiceList;
    populateVoiceList();
}

// ===============================
// Text-to-Speech Functions
// ===============================
function toggleTextToSpeech() {
    isTextToSpeechEnabled = !isTextToSpeechEnabled;
    console.log("Text-to-Speech enabled:", isTextToSpeechEnabled);
}

function speakText(text) {
    if (!isTextToSpeechEnabled || !('speechSynthesis' in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    if (voiceSelect && voiceSelect.value) {
        const voices = window.speechSynthesis.getVoices();
        const selectedVoice = voices[voiceSelect.value];
        if (selectedVoice) utterance.voice = selectedVoice;
    }
    window.speechSynthesis.speak(utterance);
}

// ===============================
// Submit Question
// ===============================
function submitQuestion() {
    console.log("Submitting question...");
    clearTimeout(submissionTimeout);

    const userId = document.getElementById('user_id')?.value || "anonymous";
    const area = document.getElementById('area')?.value || "";
    const userInput = document.getElementById('user_input')?.value || "";

    // Clear containers + reset input
    clearAllContainers();
    if (document.getElementById('user_input')) {
        document.getElementById('user_input').value = '';
    }

    fetch('/chatbot/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: userId, area: area, question: userInput })
    })
        .then(response => response.json())
        .then(data => {
            console.log("Received response from server:", data);
            console.log("blocks received:", data.blocks);

            const answerEl = document.getElementById('answer');
            if (answerEl) answerEl.innerHTML = data.answer || "";

            if (data.blocks) {
                if (data.blocks["parts-container"]) renderParts(data.blocks["parts-container"]);
                if (data.blocks["images-container"]) displayThumbnails(data.blocks["images-container"]);
                if (data.blocks["documents-container"]) displayDocuments(data.blocks["documents-container"]);
                if (data.blocks["drawings-container"]) displayDrawings(data.blocks["drawings-container"]);
            }

            if (answerEl) {
                const links = answerEl.querySelectorAll('a');
                links.forEach(link => {
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                });
            }

            if (isTextToSpeechEnabled) {
                speakText(data.answer);
            }
        })
        .catch(error => console.error('Error:', error));
}

// ===============================
// Clear All Containers
// ===============================
function clearAllContainers() {
    const containers = ["parts-container", "thumbnails-section", "doc-links-section", "drawing-section"];
    containers.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = "";
    });
}

// ===============================
// Event Bindings
// ===============================
document.addEventListener("DOMContentLoaded", () => {
    const askBtn = document.getElementById("submit-question");
    if (askBtn) askBtn.addEventListener("click", submitQuestion);

    const inputEl = document.getElementById("user_input");
    if (inputEl) {
        inputEl.addEventListener("keypress", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitQuestion();
            }
        });
    }

    const toggleBtn = document.getElementById("toggle-voice");
    if (toggleBtn) toggleBtn.addEventListener("click", toggleTextToSpeech);
    else console.log("No #toggle-voice button found, skipping voice binding.");
});
