console.log("[EMTAC] chatbot.js ANSWER_PAYLOAD_VERSION_2026_05_04_7 loaded");

// ===============================
// Chatbot Frontend Script
// Answer-first + payload-second flow
// ===============================

// Prevent duplicate answer submissions while the /ask request is active.
// This should NOT stay locked while the supporting payload loads.
let isSubmittingQuestion = false;

// Track the most recent answer request so old payload responses do not overwrite newer results.
let activeAnswerRequestId = null;

// Timer used to fade/clear the payload success visual state.
let payloadVisualClearTimer = null;

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

async function submitQuestion(event) {
    if (event && typeof event.preventDefault === "function") {
        event.preventDefault();
    }

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
    setPayloadVisualState("clear");

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
            setPayloadVisualState("error");
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
            setPayloadVisualState("loading");

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

            const hasAnyPayload = payloadHasAnyItems(answerData);

            if (hasAnyPayload) {
                setPayloadVisualState("success");
                fadePayloadVisualSuccess();
            } else {
                setPayloadVisualState("clear");
            }
        }

    } catch (error) {
        console.error("[EMTAC] Error submitting question:", error);
        setAnswerHtml("An unexpected error occurred while submitting your question.");
        setPayloadVisualState("error");

    } finally {
        // Unlock after the answer request finishes.
        // Do not wait for payload loading.
        isSubmittingQuestion = false;
    }
}

// Expose globally in case the HTML uses onclick="submitQuestion()".
window.submitQuestion = submitQuestion;


// ===============================
// Load Supporting Payload
// ===============================

async function loadSupportingPayload({ requestId, payloadEndpoint, clientType }) {
    if (!requestId) {
        console.warn("[EMTAC] Cannot load payload. Missing original request_id.");
        setPayloadLoadingMessage("");
        setPayloadVisualState("error");
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
            setPayloadVisualState("error");
            return;
        }

        if (
            payloadData.status === "invalid_input" ||
            payloadData.payload_status === "unavailable"
        ) {
            console.warn("[EMTAC] Payload unavailable:", payloadData);
            setPayloadLoadingMessage("No supporting payload was available for this answer.");
            setPayloadVisualState("error");
            return;
        }

        renderPayloadFromResponse(payloadData);
        setPayloadVisualState("success");
        fadePayloadVisualSuccess();

        console.log("[EMTAC] Payload transaction complete.", {
            request_id: payloadData.request_id || requestId,
            payload_route_request_id: payloadData.payload_route_request_id,
            payload_status: payloadData.payload_status,
            documents: payloadData.documents?.length || 0,
            parts: payloadData.parts?.length || 0,
            images: payloadData.images?.length || 0,
            drawings: payloadData.drawings?.length || 0,
        });

        // IMPORTANT:
        // Do not write "Supporting payload loaded." into the Documents panel.
        // Just clear the loading/status message after render.
        console.log("[EMTAC] Supporting payload loaded.");
        setPayloadLoadingMessage("");

    } catch (error) {
        console.error("[EMTAC] Error loading supporting payload:", error);
        setPayloadLoadingMessage("Unable to load supporting payload.");
        setPayloadVisualState("error");
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

    // Render documents first so the document panel is populated before
    // any other panel side effects.
    if (Array.isArray(payload.documents)) {
        if (payload.documents.length > 0) {
            if (typeof displayDocuments === "function") {
                displayDocuments(payload.documents);
            } else {
                console.warn("[EMTAC] displayDocuments() is not defined.");
            }
        } else {
            console.log("[EMTAC] No documents in payload.");
        }
    }

    if (Array.isArray(payload.images)) {
        if (payload.images.length > 0) {
            if (typeof displayThumbnails === "function") {
                displayThumbnails(payload.images);
            } else {
                console.warn("[EMTAC] displayThumbnails() is not defined.");
            }
        } else {
            console.log("[EMTAC] No images in payload.");
        }
    }

    if (Array.isArray(payload.parts)) {
        if (payload.parts.length > 0) {
            if (typeof renderParts === "function") {
                renderParts(payload.parts);
            } else {
                console.warn("[EMTAC] renderParts() is not defined.");
            }
        } else {
            console.log("[EMTAC] No parts in payload.");
        }
    }

    if (Array.isArray(payload.drawings)) {
        if (payload.drawings.length > 0) {
            renderDrawingsPayload(payload.drawings, data.blocks);
        } else {
            console.log("[EMTAC] No drawings in payload.");
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

function payloadHasAnyItems(data) {
    const payload = normalizePayload(data);

    return (
        (payload.documents?.length || 0) +
        (payload.parts?.length || 0) +
        (payload.images?.length || 0) +
        (payload.drawings?.length || 0)
    ) > 0;
}

function renderDrawingsPayload(drawings, blocks) {
    // Preferred drawing navigation renderer.
    // This supports Area → Model → Asset navigation from documents-container.
    if (typeof window.updateDrawingsPanelFromBlocks === "function" && blocks) {
        window.updateDrawingsPanelFromBlocks(blocks);
        return;
    }

    if (typeof displayDrawings === "function") {
        displayDrawings(drawings);
        return;
    }

    if (typeof renderDrawings === "function") {
        renderDrawings(drawings);
        return;
    }

    renderDrawingsSafely(drawings);
}

function renderDrawingsSafely(drawings) {
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
// Payload Visual Status Cue
// Theme-compatible: CSS owns colors.
// JS only toggles classes.
// ===============================

function getPayloadVisualTargets() {
    const targetIds = [
        "doc-links-section",
        "parts-container",
        "thumbnails-section",
        "drawing-section",
    ];

    const seen = new Set();

    return targetIds
        .map((id) => {
            const inner = document.getElementById(id);

            if (!inner) {
                return null;
            }

            // Prefer the full module panel perimeter if present.
            // This makes the whole container/panel glow, not just the inner content div.
            return inner.closest(".mt-panel") || inner;
        })
        .filter(Boolean)
        .filter((target) => {
            if (seen.has(target)) {
                return false;
            }

            seen.add(target);
            return true;
        });
}

function setPayloadVisualState(state) {
    if (payloadVisualClearTimer) {
        clearTimeout(payloadVisualClearTimer);
        payloadVisualClearTimer = null;
    }

    const targets = getPayloadVisualTargets();

    targets.forEach((target) => {
        target.classList.add("emtac-payload-visual");

        target.classList.remove(
            "emtac-payload-loading",
            "emtac-payload-success",
            "emtac-payload-error",
            "emtac-payload-fading"
        );

        if (state === "loading") {
            target.classList.add("emtac-payload-loading");
            return;
        }

        if (state === "success") {
            target.classList.add("emtac-payload-success");
            return;
        }

        if (state === "error") {
            target.classList.add("emtac-payload-error");
            return;
        }

        if (state === "clear") {
            target.classList.remove("emtac-payload-visual");
        }
    });
}

function fadePayloadVisualSuccess(holdMs = 2800, fadeMs = 1800) {
    if (payloadVisualClearTimer) {
        clearTimeout(payloadVisualClearTimer);
        payloadVisualClearTimer = null;
    }

    payloadVisualClearTimer = setTimeout(() => {
        const targets = getPayloadVisualTargets();

        targets.forEach((target) => {
            target.classList.add("emtac-payload-fading");
        });

        payloadVisualClearTimer = setTimeout(() => {
            payloadVisualClearTimer = null;
            setPayloadVisualState("clear");
        }, fadeMs);

    }, holdMs);
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

    // IMPORTANT:
    // Do NOT write status messages into #doc-links-section.
    // That is the actual Documents panel and would overwrite rendered chunks.
    if (message) {
        console.log("[EMTAC] Payload status:", message);
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

// Expose for other partials if needed.
window.escapeHtml = window.escapeHtml || escapeHtml;


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
        askBtn.addEventListener("click", (event) => {
            event.preventDefault();
            submitQuestion(event);
        });
    }

    const inputEl = document.getElementById("user_input");

    if (inputEl) {
        inputEl.addEventListener("keypress", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitQuestion(event);
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