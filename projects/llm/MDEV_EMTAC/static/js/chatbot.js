console.log("[EMTAC] chatbot.js ANSWER_PAYLOAD_VERSION_2026_05_04_3 loaded");
// ===============================
// Chatbot Frontend Script
// Answer-first + payload-second flow
// ===============================

// Prevent duplicate answer submissions while the /ask request is active.
// This should NOT stay locked while the supporting payload loads.
let isSubmittingQuestion = false;

// Track the most recent answer request so old payload responses do not overwrite newer results.
let activeAnswerRequestId = null;

// Text-to-speech state
let isTextToSpeechEnabled = false;
let voiceSelect;


// ===============================
// Voice Setup
// ===============================

function populateVoiceList() {
    if (!("speechSynthesis" in window)) return;

    if (!voiceSelect) {
        voiceSelect = document.getElementById("voice-select");

        if (!voiceSelect) {
            console.warn("[EMTAC] No #voice-select element found, skipping voice list population.");
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

if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = populateVoiceList;
    populateVoiceList();
}


// ===============================
// Text-to-Speech Functions
// ===============================

function toggleTextToSpeech() {
    isTextToSpeechEnabled = !isTextToSpeechEnabled;
    console.log("[EMTAC] Text-to-Speech enabled:", isTextToSpeechEnabled);
}

function speakText(text) {
    if (!isTextToSpeechEnabled || !("speechSynthesis" in window)) return;

    const utterance = new SpeechSynthesisUtterance(text || "");

    if (voiceSelect && voiceSelect.value) {
        const voices = window.speechSynthesis.getVoices();
        const selectedVoice = voices[voiceSelect.value];

        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
    }

    window.speechSynthesis.speak(utterance);
}


// ===============================
// Submit Question
// ===============================

async function submitQuestion() {
    if (isSubmittingQuestion) {
        console.warn("[EMTAC] Question already submitting. Ignoring duplicate submit.");
        return;
    }

    console.log("[EMTAC] Submitting question...");

    const userId = document.getElementById("user_id")?.value || "anonymous";
    const area = document.getElementById("area")?.value || "";
    const inputEl = document.getElementById("user_input");
    const userInput = inputEl?.value || "";

    if (!userInput.trim()) {
        console.warn("[EMTAC] Empty question. Nothing submitted.");
        return;
    }

    isSubmittingQuestion = true;
    activeAnswerRequestId = null;

    clearAllContainers();
    setAnswerHtml("Thinking...");
    setPayloadLoadingMessage("");

    if (inputEl) {
        inputEl.value = "";
    }

    try {
        // ------------------------------------------------------
        // 1. Ask route: answer first
        // ------------------------------------------------------
        const answerResponse = await fetch("/chatbot/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                userId: userId,
                area: area,
                question: userInput,
                clientType: "web",
            }),
        });

        const answerData = await safeJsonResponse(answerResponse);

        console.log("[EMTAC] Answer response:", answerData);

        console.log("[EMTAC] Payload trigger check:", {
            shouldLoadPayload: shouldLoadPayload(answerData),
            payload_status: answerData.payload_status,
            request_id: answerData.request_id,
            payload_endpoint: answerData.payload_endpoint,
        });

        if (!answerResponse.ok || answerData.status === "error") {
            setAnswerHtml(answerData.answer || "An unexpected error occurred.");
            return;
        }

        activeAnswerRequestId = answerData.request_id || null;

        // Render answer immediately.
        setAnswerHtml(answerData.answer || "");
        applyAnswerLinkBehavior();

        if (isTextToSpeechEnabled) {
            speakText(answerData.answer || "");
        }

        // ------------------------------------------------------
        // 2. Payload route: supporting UI payload second
        // ------------------------------------------------------
        // Do NOT await this.
        // The answer has already been rendered, and the payload can load
        // independently without blocking another question submission.
        if (shouldLoadPayload(answerData)) {
            console.log("[EMTAC] Triggering supporting payload request now.", {
                requestId: answerData.request_id,
                payloadEndpoint: answerData.payload_endpoint,
            });

            setPayloadLoadingMessage("Loading related documents, images, parts, and drawings...");

            loadSupportingPayload({
                requestId: answerData.request_id,
                payloadEndpoint: answerData.payload_endpoint,
                clientType: "web",
            });
        } else {
            console.log("[EMTAC] No payload request needed.", {
                payload_status: answerData.payload_status,
                request_id: answerData.request_id,
            });

            // Backward compatibility:
            // If the answer route still returned payload data, render it.
            renderPayloadFromResponse(answerData);
        }

    } catch (error) {
        console.error("[EMTAC] Error submitting question:", error);
        setAnswerHtml("An unexpected error occurred while submitting your question.");

    } finally {
        // Unlock after the answer request finishes.
        // Do not wait for payload loading.
        isSubmittingQuestion = false;
    }
}


// ===============================
// Load Supporting Payload
// ===============================

async function loadSupportingPayload({ requestId, payloadEndpoint, clientType }) {
    if (!requestId) {
        console.warn("[EMTAC] Cannot load payload. Missing original request_id.");
        setPayloadLoadingMessage("");
        return;
    }

    const endpoint = resolvePayloadEndpoint(payloadEndpoint);

    console.log("[EMTAC] Starting payload transaction.", {
        endpoint: endpoint,
        requestId: requestId,
        clientType: clientType || "web",
    });

    try {
        const payloadResponse = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                requestId: requestId,
                clientType: clientType || "web",
            }),
        });

        const payloadData = await safeJsonResponse(payloadResponse);

        console.log("[EMTAC] Payload response:", payloadData);

        // Prevent stale payloads from an older question from overwriting
        // the current panels.
        if (activeAnswerRequestId && requestId !== activeAnswerRequestId) {
            console.warn("[EMTAC] Ignoring stale payload response.", {
                payloadRequestId: requestId,
                activeAnswerRequestId: activeAnswerRequestId,
            });
            return;
        }

        if (
            !payloadResponse.ok ||
            payloadData.status === "error" ||
            payloadData.payload_status === "error"
        ) {
            console.error("[EMTAC] Payload load failed:", payloadData);
            setPayloadLoadingMessage("Unable to load supporting payload.");
            return;
        }

        if (
            payloadData.status === "invalid_input" ||
            payloadData.payload_status === "unavailable"
        ) {
            console.warn("[EMTAC] Payload unavailable:", payloadData);
            setPayloadLoadingMessage("No supporting payload was available for this answer.");
            return;
        }

        renderPayloadFromResponse(payloadData);

        console.log("[EMTAC] Payload transaction complete.", {
            request_id: payloadData.request_id || requestId,
            payload_route_request_id: payloadData.payload_route_request_id,
            payload_status: payloadData.payload_status,
            documents: payloadData.documents?.length || 0,
            parts: payloadData.parts?.length || 0,
            images: payloadData.images?.length || 0,
            drawings: payloadData.drawings?.length || 0,
        });

        setPayloadLoadingMessage("Supporting payload loaded.");
        setTimeout(() => {
            setPayloadLoadingMessage("");
        }, 2500);

    } catch (error) {
        console.error("[EMTAC] Error loading supporting payload:", error);
        setPayloadLoadingMessage("Unable to load supporting payload.");
    }
}


// ===============================
// Payload Rendering
// ===============================

function renderPayloadFromResponse(data) {
    if (!data || typeof data !== "object") {
        console.warn("[EMTAC] No payload data to render.");
        return;
    }

    const payload = normalizePayload(data);

    console.log("[EMTAC] Normalized payload:", payload);

    if (Array.isArray(payload.parts)) {
        if (payload.parts.length > 0) {
            if (typeof renderParts === "function") {
                renderParts(payload.parts);
            } else {
                console.warn("[EMTAC] renderParts() is not defined.");
            }
        }
    }

    if (Array.isArray(payload.images)) {
        if (payload.images.length > 0) {
            if (typeof displayThumbnails === "function") {
                displayThumbnails(payload.images);
            } else {
                console.warn("[EMTAC] displayThumbnails() is not defined.");
            }
        }
    }

    if (Array.isArray(payload.documents)) {
        if (payload.documents.length > 0) {
            if (typeof displayDocuments === "function") {
                displayDocuments(payload.documents);
            } else {
                console.warn("[EMTAC] displayDocuments() is not defined.");
            }
        }
    }

    if (Array.isArray(payload.drawings)) {
        if (payload.drawings.length > 0) {
            renderDrawingsSafely(payload.drawings);
        }
    }

    const totalItems =
        (payload.documents?.length || 0) +
        (payload.parts?.length || 0) +
        (payload.images?.length || 0) +
        (payload.drawings?.length || 0);

    if (totalItems === 0) {
        console.log("[EMTAC] Payload loaded but contained no supporting items.");
    }
}

function normalizePayload(data) {
    const blocks = data.blocks && typeof data.blocks === "object"
        ? data.blocks
        : {};

    return {
        documents:
            data.documents ||
            blocks["documents-container"] ||
            blocks.documents ||
            [],

        parts:
            data.parts ||
            blocks["parts-container"] ||
            blocks.parts ||
            [],

        images:
            data.images ||
            blocks["images-container"] ||
            blocks.thumbnails ||
            blocks.images ||
            [],

        drawings:
            data.drawings ||
            blocks["drawings-container"] ||
            blocks.drawings ||
            [],
    };
}

function renderDrawingsSafely(drawings) {
    if (typeof displayDrawings === "function") {
        displayDrawings(drawings);
        return;
    }

    if (typeof renderDrawings === "function") {
        renderDrawings(drawings);
        return;
    }

    const drawingSection = document.getElementById("drawing-section");

    if (!drawingSection) {
        console.warn("[EMTAC] No drawing renderer found and #drawing-section does not exist.");
        return;
    }

    drawingSection.innerHTML = "";

    drawings.forEach((drawing) => {
        const link = document.createElement("a");

        const title =
            drawing.title ||
            drawing.drw_name ||
            drawing.name ||
            drawing.drw_number ||
            "Drawing";

        const href =
            drawing.url ||
            drawing.file_url ||
            drawing.file_path ||
            "#";

        link.textContent = title;
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.classList.add("drawing-link");

        const wrapper = document.createElement("div");
        wrapper.classList.add("drawing-link-wrapper");
        wrapper.appendChild(link);

        drawingSection.appendChild(wrapper);
    });
}


// ===============================
// Helpers
// ===============================

async function safeJsonResponse(response) {
    try {
        return await response.json();
    } catch (error) {
        console.error("[EMTAC] Failed to parse JSON response:", error);
        return {
            status: "error",
            payload_status: "error",
            answer: "Invalid server response.",
            message: "Invalid server response.",
        };
    }
}

function shouldLoadPayload(answerData) {
    if (!answerData || typeof answerData !== "object") {
        return false;
    }

    return (
        answerData.payload_status === "pending" &&
        Boolean(answerData.request_id)
    );
}

function resolvePayloadEndpoint(payloadEndpoint) {
    if (!payloadEndpoint) {
        return "/chatbot/ask/payload";
    }

    // Backend may return "/ask/payload", but this blueprint is mounted under "/chatbot".
    if (payloadEndpoint === "/ask/payload") {
        return "/chatbot/ask/payload";
    }

    // If backend already returns full blueprint path, use it.
    if (payloadEndpoint === "/chatbot/ask/payload") {
        return payloadEndpoint;
    }

    // Do not allow unknown/external endpoint values from response payload.
    return "/chatbot/ask/payload";
}

function setAnswerHtml(html) {
    const answerEl = document.getElementById("answer");

    if (!answerEl) {
        console.warn("[EMTAC] #answer element not found.");
        return;
    }

    answerEl.innerHTML = html || "";
}

function applyAnswerLinkBehavior() {
    const answerEl = document.getElementById("answer");

    if (!answerEl) return;

    const links = answerEl.querySelectorAll("a");

    links.forEach((link) => {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
    });
}

function setPayloadLoadingMessage(message) {
    const possibleTargets = [
        "payload-loading-message",
        "supporting-payload-status",
    ];

    for (const id of possibleTargets) {
        const el = document.getElementById(id);

        if (el) {
            el.innerHTML = message
                ? `<p class="payload-loading">${escapeHtml(message)}</p>`
                : "";
            return;
        }
    }

    // Fallback:
    // Only write into doc-links-section if we are setting a message.
    // Do not clear doc-links-section here because that could remove loaded documents.
    if (message) {
        const docSection = document.getElementById("doc-links-section");

        if (docSection) {
            docSection.innerHTML = `<p class="payload-loading">${escapeHtml(message)}</p>`;
            return;
        }

        console.log("[EMTAC]", message);
    }
}

function escapeHtml(value) {
    return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


// ===============================
// Clear All Containers
// ===============================

function clearAllContainers() {
    const containers = [
        "parts-container",
        "thumbnails-section",
        "doc-links-section",
        "drawing-section",
        "payload-loading-message",
        "supporting-payload-status",
    ];

    containers.forEach((id) => {
        const el = document.getElementById(id);

        if (el) {
            el.innerHTML = "";
        }
    });
}


// ===============================
// Event Bindings
// ===============================

document.addEventListener("DOMContentLoaded", () => {
    const askBtn = document.getElementById("submit-question");

    if (askBtn) {
        askBtn.addEventListener("click", submitQuestion);
    }

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

    if (toggleBtn) {
        toggleBtn.addEventListener("click", toggleTextToSpeech);
    } else {
        console.log("[EMTAC] No #toggle-voice button found, skipping voice binding.");
    }
});