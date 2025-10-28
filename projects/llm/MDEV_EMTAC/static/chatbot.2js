// Check if the browser supports SpeechRecognition
var desiredVoiceName = "Microsoft Aria Online (Natural) - English (United States) (en-US)";
var voiceSelect = document.getElementById('voice-selection');
var availableVoices = [];
var submissionTimeout;
var timeoutDuration = 1750; // 1.75 seconds of inactivity

// Helper function to select the desired voice
function selectDesiredVoice() {
    for (var i = 0; i < voiceSelect.options.length; i++) {
        if (voiceSelect.options[i].textContent.includes(desiredVoiceName)) {
            voiceSelect.selectedIndex = i;
            break;
        }
    }
}

window.speechSynthesis.onvoiceschanged = function() {
    availableVoices = window.speechSynthesis.getVoices();
    populateVoiceList();
    selectDesiredVoice(); // Call the function to select the desired voice
};

function populateVoiceList() {
    voiceSelect.innerHTML = ''; // Clear existing options
    availableVoices.forEach(voice => {
        var option = document.createElement('option');
        option.textContent = voice.name + ' (' + voice.lang + ')';
        
        option.setAttribute('data-name', voice.name);
        voiceSelect.appendChild(option);
    });
}

function speakText(text) {
    var selectedVoiceName = voiceSelect.selectedOptions[0].getAttribute('data-name');
    var selectedVoice = availableVoices.find(voice => voice.name === selectedVoiceName);

    if ('speechSynthesis' in window && isTextToSpeechEnabled) { // Check if text-to-speech is enabled
        // Stop speech recognition before speaking
        if (isListening) {
            speechRecognizer.stop();
        }

        var utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = selectedVoice;
        utterance.pitch = 0.8; // Adjusted pitch
        utterance.rate = 1.; // Adjusted rate
        utterance.volume = 1; // Ensure this is within the range [0, 1]

        utterance.onend = function() {
            // Optionally restart speech recognition after speaking
            if (isListening) {
                speechRecognizer.start();
            }
        };

        window.speechSynthesis.speak(utterance);
    } else {
        console.log("Your browser does not support text-to-speech or text-to-speech is disabled.");
    }
}

// Boolean variable to track text-to-speech state
var isTextToSpeechEnabled = false;

// Function to toggle text-to-speech
function toggleTextToSpeech() {
    isTextToSpeechEnabled = !isTextToSpeechEnabled;
    var toggleButton = document.getElementById('toggle-text-to-speech'); // Updated ID
    
    if (isTextToSpeechEnabled) {
        toggleButton.textContent = 'Disable Text-to-Speech';
    } else {
        toggleButton.textContent = 'Enable Text-to-Speech';
    }
}

// Event listener for the text-to-speech toggle button
document.getElementById('toggle-text-to-speech').addEventListener('click', toggleTextToSpeech);

if ('webkitSpeechRecognition' in window) {
    var speechRecognizer = new webkitSpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = 'en-US';

    var isListening = false;
    var toggleVoiceButton = document.getElementById('toggle-voice');

    function startVoiceRecognition() {
        speechRecognizer.start();
        toggleVoiceButton.textContent = 'Disable Voice Recognition';
        isListening = true;
    }

    function stopVoiceRecognition() {
        speechRecognizer.stop();
        toggleVoiceButton.textContent = 'Enable Voice Recognition';
        isListening = false;
    }

    toggleVoiceButton.addEventListener('click', function () {
        if (isListening) {
            stopVoiceRecognition();
        } else {
            startVoiceRecognition();
        }
    });

	var accumulatedTranscript = ''; // Accumulate transcript across results
	var isFinalized = false; // Track if the last result was finalized

speechRecognizer.onresult = function (event) {
    var interimTranscript = ''; // Temporary storage for the current non-final result
    var currentResultIsFinal = false;

    for (var i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
            accumulatedTranscript += event.results[i][0].transcript + ' '; // Add final result to accumulated transcript
            currentResultIsFinal = true;
            isFinalized = true; // Mark as finalized
        } else {
            interimTranscript += event.results[i][0].transcript; // Collect interim results
            isFinalized = false; // Still collecting, not finalized
        }
    }

    // If there's interim transcript, use it to update the UI immediately
    if (!isFinalized && interimTranscript.trim() !== '') {
        document.getElementById('user_input').value = accumulatedTranscript + interimTranscript;
    } else if (currentResultIsFinal) {
        document.getElementById('user_input').value = accumulatedTranscript;
    }

    // Reset the submission timeout on every result
    clearTimeout(submissionTimeout);
    submissionTimeout = setTimeout(function() {
        if (isFinalized) {
            submitQuestion(); // Only auto-submit if the last result was finalized
        }
    }, timeoutDuration);
};


    speechRecognizer.onerror = function (event) {
        console.error('Speech recognition error', event);
        stopVoiceRecognition();
    };
} else {
    console.log("Your browser does not support Speech Recognition.");
    document.getElementById('toggle-voice').style.display = 'none';
}

function clearThumbnails() {
    const thumbnailsSection = document.getElementById("thumbnails-section");
    thumbnailsSection.innerHTML = ""; // Clear existing thumbnails
}




function submitQuestion() {
    console.log("Submitting question...");
    clearTimeout(submissionTimeout);
    var userId = document.getElementById('user_id').value;
    var area = document.getElementById('area').value;
    var userInput = document.getElementById('user_input').value;

	// Clear thumbnails before submitting the question
    clearThumbnails();
	
    fetch('/chatbot/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId: userId, area: area, question: userInput })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received response from server:", data);
        // Insert the response into the innerHTML of the answer element
        document.getElementById('answer').innerHTML = data.answer;

        // Display thumbnails if available
        if (Array.isArray(data.answer) && data.answer.length > 0) {
            displayThumbnails(data.answer);
        }

        // After the response is inserted, modify all <a> tags to open in a new tab
        var links = document.getElementById('answer').querySelectorAll('a');
        links.forEach(function(link) {
            link.target = "_blank"; // Make links open in a new tab
            link.rel = "noopener noreferrer"; // Security enhancement
        });

        // Speak the answer text if text-to-speech is enabled
        if (isTextToSpeechEnabled) {
            speakText(data.answer);
        }
    })
    .catch(error => console.error('Error:', error));
}

document.getElementById('submit-question').addEventListener('click', submitQuestion);
