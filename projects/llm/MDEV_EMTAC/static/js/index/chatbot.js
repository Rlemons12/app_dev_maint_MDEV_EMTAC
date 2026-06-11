console.log("[EMTAC] chatbot.js ANSWER_PAYLOAD_VERSION_2026_06_11_CONVERSATION_MODE_TOGGLE loaded");

// ===============================
// Chatbot Frontend Script
// Answer-first + payload-second flow
// Conversation ON/OFF support
// ===============================

// Prevent duplicate answer submissions while the /ask request is active.
// This should NOT stay locked while the supporting payload loads.
let isSubmittingQuestion = false;

// Track the most recent answer request so old payload responses do not overwrite newer results.
let activeAnswerRequestId = null;

// Track the active conversational-memory session.
// Backend returns this as conversation_id from /chatbot/ask.
// Frontend must send it back on the next /chatbot/ask request when Conversation is ON.
const EMTAC_CONVERSATION_STORAGE_KEY = "emtac_active_conversation_id";
const EMTAC_CONVERSATION_ENABLED_STORAGE_KEY = "emtac_conversation_enabled";

let activeConversationId = restoreConversationIdFromSessionStorage();
let isConversationMemoryEnabled = restoreConversationEnabledFromLocalStorage();

if (!isConversationMemoryEnabled) {
    activeConversationId = null;
    persistConversationIdToSessionStorage(null);
}

syncConversationGlobals(activeConversationId);
syncConversationModeGlobals(isConversationMemoryEnabled);

// Timer used to fade/clear the payload success visual state.
let payloadVisualClearTimer = null;

// Text-to-speech state
let isTextToSpeechEnabled = false;
let voiceSelect;


// ===============================
// Conversation Mode Helpers
// ===============================

function restoreConversationEnabledFromLocalStorage() {
    try {
        const storedValue = localStorage.getItem(EMTAC_CONVERSATION_ENABLED_STORAGE_KEY);

        if (storedValue === null || storedValue === undefined) {
            return true;
        }

        return storedValue === "true";
    } catch (error) {
        console.debug("[EMTAC] Unable to restore conversation_enabled from localStorage.", error);
        return true;
    }
}

function persistConversationEnabledToLocalStorage(enabled) {
    try {
        localStorage.setItem(EMTAC_CONVERSATION_ENABLED_STORAGE_KEY, String(Boolean(enabled)));
    } catch (error) {
        console.debug("[EMTAC] Unable to persist conversation_enabled to localStorage.", error);
    }
}

function syncConversationModeGlobals(enabled) {
    const normalizedEnabled = Boolean(enabled);

    window.EMTAC_CONVERSATION_ENABLED = normalizedEnabled;
    window.emtacConversationEnabled = normalizedEnabled;
    window.five331ConversationEnabled = normalizedEnabled;
    window.conversationEnabled = normalizedEnabled;
    window.conversationMode = normalizedEnabled ? "conversation" : "single_turn";

    const enabledInput = document.getElementById("conversation_enabled");
    const modeInput = document.getElementById("conversation_mode");

    if (enabledInput) {
        enabledInput.value = normalizedEnabled ? "true" : "false";
    }

    if (modeInput) {
        modeInput.value = normalizedEnabled ? "conversation" : "single_turn";
    }
}

function isConversationEnabledForAsk() {
    /*
      Prefer the Quick Actions partial helper when present.
      Fallback to this file's own localStorage-backed state.
    */

    try {
        if (typeof window.isFive331ConversationEnabled === "function") {
            isConversationMemoryEnabled = Boolean(window.isFive331ConversationEnabled());
            syncConversationModeGlobals(isConversationMemoryEnabled);
            return isConversationMemoryEnabled;
        }
    } catch (error) {
        console.debug("[EMTAC] Unable to read five331 conversation mode helper.", error);
    }

    isConversationMemoryEnabled = Boolean(isConversationMemoryEnabled);
    syncConversationModeGlobals(isConversationMemoryEnabled);
    return isConversationMemoryEnabled;
}

function setEMTACConversationEnabled(enabled, options = {}) {
    const normalizedEnabled = Boolean(enabled);

    isConversationMemoryEnabled = normalizedEnabled;
    persistConversationEnabledToLocalStorage(normalizedEnabled);
    syncConversationModeGlobals(normalizedEnabled);

    if (!normalizedEnabled) {
        clearActiveConversationId();
    }

    /*
      Keep the Quick Actions button in sync when that partial exists.
      Avoid requiring the partial to exist for this JS to work.
    */
    if (
        options.syncFive331 !== false &&
        typeof window.setFive331ConversationEnabled === "function"
    ) {
        try {
            window.setFive331ConversationEnabled(normalizedEnabled, {
                log: false,
            });
        } catch (error) {
            console.debug("[EMTAC] Unable to sync five331 conversation mode button.", error);
        }
    }

    console.info("[EMTAC] Conversation mode set:", {
        conversation_enabled: normalizedEnabled,
        conversation_mode: normalizedEnabled ? "conversation" : "single_turn",
        active_conversation_id: getActiveConversationId(),
    });
}

function getConversationPayloadPatchForAsk() {
    /*
      The Quick Actions partial exposes getFive331ConversationPayloadPatch().
      Use it when present so the button state is the source of truth.

      Returned shape:
        {
          conversation_enabled: boolean,
          conversation_mode: "conversation" | "single_turn",
          conversation_id: uuid | null
        }
    */

    try {
        if (typeof window.getFive331ConversationPayloadPatch === "function") {
            const patch = window.getFive331ConversationPayloadPatch();

            const enabled = Boolean(patch && patch.conversation_enabled);

            isConversationMemoryEnabled = enabled;
            persistConversationEnabledToLocalStorage(enabled);
            syncConversationModeGlobals(enabled);

            if (!enabled) {
                clearActiveConversationId();

                return {
                    conversation_enabled: false,
                    conversation_mode: "single_turn",
                    conversation_id: null,
                };
            }

            const patchConversationId =
                patch && patch.conversation_id
                    ? String(patch.conversation_id).trim()
                    : getActiveConversationId();

            return {
                conversation_enabled: true,
                conversation_mode: "conversation",
                conversation_id: patchConversationId || null,
            };
        }
    } catch (error) {
        console.debug("[EMTAC] Unable to use five331 conversation payload patch.", error);
    }

    const enabled = isConversationEnabledForAsk();

    if (!enabled) {
        clearActiveConversationId();

        return {
            conversation_enabled: false,
            conversation_mode: "single_turn",
            conversation_id: null,
        };
    }

    return {
        conversation_enabled: true,
        conversation_mode: "conversation",
        conversation_id: getActiveConversationId(),
    };
}

window.setEMTACConversationEnabled = setEMTACConversationEnabled;
window.isEMTACConversationEnabled = isConversationEnabledForAsk;
window.getEMTACConversationPayloadPatchForAsk = getConversationPayloadPatchForAsk;

window.addEventListener("emtac:conversation-mode-changed", function (event) {
    const detail = event && event.detail ? event.detail : {};
    const enabled = Boolean(detail.conversation_enabled);

    isConversationMemoryEnabled = enabled;
    persistConversationEnabledToLocalStorage(enabled);
    syncConversationModeGlobals(enabled);

    if (!enabled) {
        clearActiveConversationId();
    }

    console.info("[EMTAC] Conversation mode changed event received:", {
        conversation_enabled: enabled,
        conversation_mode: enabled ? "conversation" : "single_turn",
    });
});


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

    const conversationPatch = getConversationPayloadPatchForAsk();
    const outgoingConversationId = conversationPatch.conversation_id || null;
    const outgoingDocumentScope = getActiveDocumentScopeForAsk();

    console.log("[EMTAC] Conversation memory state before ask:", {
        conversation_enabled: conversationPatch.conversation_enabled,
        conversation_mode: conversationPatch.conversation_mode,
        conversation_id: outgoingConversationId,
        hasConversation: Boolean(outgoingConversationId),
    });

    console.log("[EMTAC DOCUMENT MODE] Document scope before ask:", {
        document_scope: outgoingDocumentScope,
        enabled: Boolean(outgoingDocumentScope && outgoingDocumentScope.enabled),
        scope_type: outgoingDocumentScope?.scope_type || null,
        complete_document_id: outgoingDocumentScope?.complete_document_id || null,
        document_name: outgoingDocumentScope?.document_name || null,
    });

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
        const askRequestBody = {
            userId: userId,
            area: area,
            question: userInput,
            clientType: "web",

            // Conversation ON/OFF mode.
            conversation_enabled: conversationPatch.conversation_enabled,
            conversation_mode: conversationPatch.conversation_mode,
            conversation_id: outgoingConversationId,

            // Document-scoped conversation mode.
            // Null when no document mode is active.
            document_scope: outgoingDocumentScope,
        };

        console.log("[EMTAC] Ask request body:", {
            userId: askRequestBody.userId,
            area: askRequestBody.area,
            clientType: askRequestBody.clientType,
            conversation_enabled: askRequestBody.conversation_enabled,
            conversation_mode: askRequestBody.conversation_mode,
            conversation_id: askRequestBody.conversation_id,
            has_document_scope: Boolean(askRequestBody.document_scope),
            document_scope: askRequestBody.document_scope,
        });

        const answerResponse = await fetch("/chatbot/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(askRequestBody),
        });

        const answerData = await safeJsonResponse(answerResponse);

        console.log("[EMTAC] Answer response:", answerData);

        console.log("[EMTAC] Payload trigger check:", {
            shouldLoadPayload: shouldLoadPayload(answerData),
            payload_status: answerData.payload_status,
            request_id: answerData.request_id,
            payload_endpoint: answerData.payload_endpoint,
            conversation_enabled: askRequestBody.conversation_enabled,
            conversation_mode: askRequestBody.conversation_mode,
            conversation_id: extractConversationId(answerData),
            document_scope_sent: outgoingDocumentScope,
        });

        const returnedConversationId = extractConversationId(answerData);

        if (askRequestBody.conversation_enabled && returnedConversationId) {
            setActiveConversationId(returnedConversationId);
        }

        if (!askRequestBody.conversation_enabled) {
            clearActiveConversationId();
        }

        if (answerData.status === "session_ended") {
            clearActiveConversationId();
        }

        if (!answerResponse.ok || answerData.status === "error") {
            setAnswerHtml(answerData.answer || "An unexpected error occurred.");
            setPayloadVisualState("error");
            return;
        }

        activeAnswerRequestId = answerData.request_id || null;

        // Render answer immediately.
        setAnswerHtml(answerData.answer || "");
        applyAnswerLinkBehavior();

        // Store the latest Q&A context so update_qanda_table.js can submit
        // rating/comment feedback to /chatbot/update_qanda.
        const activeFeedbackConversationId = askRequestBody.conversation_enabled
            ? getActiveConversationId()
            : null;

        if (typeof window.updateQandAFeedbackContext === "function") {
            window.updateQandAFeedbackContext({
                userId: userId,
                question: userInput,
                answer: answerData.answer || "",
                requestId: answerData.request_id || null,
                conversationId: activeFeedbackConversationId,
                conversation_id: activeFeedbackConversationId,

                conversationEnabled: askRequestBody.conversation_enabled,
                conversation_enabled: askRequestBody.conversation_enabled,
                conversationMode: askRequestBody.conversation_mode,
                conversation_mode: askRequestBody.conversation_mode,

                // Helpful for future debugging/auditing.
                documentScope: outgoingDocumentScope,
                document_scope: outgoingDocumentScope,
            });
        } else {
            console.warn("[EMTAC] updateQandAFeedbackContext() is not available yet.");
        }

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
                conversationEnabled: askRequestBody.conversation_enabled,
                conversationMode: askRequestBody.conversation_mode,
                conversationId: askRequestBody.conversation_enabled ? getActiveConversationId() : null,
                documentScope: outgoingDocumentScope,
            });

            setPayloadLoadingMessage("Loading related documents, images, parts, and drawings...");
            setPayloadVisualState("loading");

            loadSupportingPayload({
                requestId: answerData.request_id,
                payloadEndpoint: answerData.payload_endpoint,
                clientType: "web",
                conversationEnabled: askRequestBody.conversation_enabled,
                conversationMode: askRequestBody.conversation_mode,
                conversationId: askRequestBody.conversation_enabled ? getActiveConversationId() : null,
            });
        } else {
            console.log("[EMTAC] No payload request needed.", {
                payload_status: answerData.payload_status,
                request_id: answerData.request_id,
                conversation_enabled: askRequestBody.conversation_enabled,
                conversation_mode: askRequestBody.conversation_mode,
                conversation_id: askRequestBody.conversation_enabled ? getActiveConversationId() : null,
                document_scope: outgoingDocumentScope,
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

async function loadSupportingPayload({
    requestId,
    payloadEndpoint,
    clientType,
    conversationId,
    conversationEnabled,
    conversationMode,
}) {
    if (!requestId) {
        console.warn("[EMTAC] Cannot load payload. Missing original request_id.");
        setPayloadLoadingMessage("");
        setPayloadVisualState("error");
        return;
    }

    const endpoint = resolvePayloadEndpoint(payloadEndpoint);
    const enabled = conversationEnabled !== undefined
        ? Boolean(conversationEnabled)
        : isConversationEnabledForAsk();

    const mode = conversationMode || (enabled ? "conversation" : "single_turn");
    const payloadConversationId = enabled
        ? (conversationId || getActiveConversationId())
        : null;

    console.log("[EMTAC] Starting payload transaction.", {
        endpoint: endpoint,
        requestId: requestId,
        clientType: clientType || "web",
        conversation_enabled: enabled,
        conversation_mode: mode,
        conversationId: payloadConversationId,
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
                conversation_enabled: enabled,
                conversation_mode: mode,
                conversation_id: payloadConversationId,
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
            conversation_enabled: enabled,
            conversation_mode: mode,
            conversation_id: enabled
                ? (extractConversationId(payloadData) || payloadConversationId || getActiveConversationId())
                : null,
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
// Debuggable + guarded rendering
// ===============================

// Temporary front-end render limits.
// This prevents the browser from trying to draw 883 parts and 1272 drawings at once.
// Set either value to 0 or Infinity if you want full rendering again.
const EMTAC_RENDER_LIMITS = {
    documents: Infinity,
    images: Infinity,
    parts: 100,
    drawings: 100,
};

// Keep last full payload available in browser console.
// You can inspect it with:
// window.EMTAC_LAST_PAYLOAD
window.EMTAC_LAST_PAYLOAD = null;

function renderPayloadFromResponse(data) {
    if (!data || typeof data !== "object") {
        console.warn("[EMTAC] No payload data to render.");
        return;
    }

    console.time("[EMTAC] payload render total");

    const payload = normalizePayload(data);

    window.EMTAC_LAST_PAYLOAD = {
        raw: data,
        normalized: payload,
        receivedAt: new Date().toISOString(),
    };

    const counts = {
        documents: payload.documents?.length || 0,
        images: payload.images?.length || 0,
        parts: payload.parts?.length || 0,
        drawings: payload.drawings?.length || 0,
        hasBlocks: Boolean(data.blocks),
        hasDrawingBlockRenderer: typeof window.updateDrawingsPanelFromBlocks === "function",
    };

    console.log("[EMTAC] Normalized payload:", payload);
    console.log("[EMTAC] Payload counts before render:", counts);

    const renderPayload = {
        documents: limitPayloadArray(payload.documents, EMTAC_RENDER_LIMITS.documents),
        images: limitPayloadArray(payload.images, EMTAC_RENDER_LIMITS.images),
        parts: limitPayloadArray(payload.parts, EMTAC_RENDER_LIMITS.parts),
        drawings: limitPayloadArray(payload.drawings, EMTAC_RENDER_LIMITS.drawings),
    };

    console.log("[EMTAC] Payload counts being rendered:", {
        documents: renderPayload.documents.length,
        images: renderPayload.images.length,
        parts: renderPayload.parts.length,
        drawings: renderPayload.drawings.length,
    });

    // -----------------------------
    // Documents
    // -----------------------------
    console.time("[EMTAC] render documents");

    if (Array.isArray(renderPayload.documents)) {
        if (renderPayload.documents.length > 0) {
            if (typeof displayDocuments === "function") {
                displayDocuments(renderPayload.documents);
                appendPayloadLimitNotice(
                    "doc-links-section",
                    "documents",
                    renderPayload.documents.length,
                    payload.documents.length
                );
            } else {
                console.warn("[EMTAC] displayDocuments() is not defined.");
            }
        } else {
            console.log("[EMTAC] No documents in payload.");
        }
    }

    console.timeEnd("[EMTAC] render documents");

    // -----------------------------
    // Images
    // -----------------------------
    console.time("[EMTAC] render images");

    if (Array.isArray(renderPayload.images)) {
        if (renderPayload.images.length > 0) {
            if (typeof displayThumbnails === "function") {
                displayThumbnails(renderPayload.images);
                appendPayloadLimitNotice(
                    "thumbnails-section",
                    "images",
                    renderPayload.images.length,
                    payload.images.length
                );
            } else {
                console.warn("[EMTAC] displayThumbnails() is not defined.");
            }
        } else {
            console.log("[EMTAC] No images in payload.");
        }
    }

    console.timeEnd("[EMTAC] render images");

    // -----------------------------
    // Parts
    // -----------------------------
    console.time("[EMTAC] render parts");

    if (Array.isArray(renderPayload.parts)) {
        if (renderPayload.parts.length > 0) {
            if (typeof renderParts === "function") {
                renderParts(renderPayload.parts);
                appendPayloadLimitNotice(
                    "parts-container",
                    "parts",
                    renderPayload.parts.length,
                    payload.parts.length
                );
            } else {
                console.warn("[EMTAC] renderParts() is not defined.");
            }
        } else {
            console.log("[EMTAC] No parts in payload.");
        }
    }

    console.timeEnd("[EMTAC] render parts");

    // -----------------------------
    // Drawings
    // -----------------------------
    console.time("[EMTAC] render drawings");

    if (Array.isArray(renderPayload.drawings)) {
        if (renderPayload.drawings.length > 0) {
            renderDrawingsPayload(renderPayload.drawings, data.blocks, {
                renderedCount: renderPayload.drawings.length,
                totalCount: payload.drawings.length,
            });

            appendPayloadLimitNotice(
                "drawing-section",
                "drawings",
                renderPayload.drawings.length,
                payload.drawings.length
            );
        } else {
            console.log("[EMTAC] No drawings in payload.");
        }
    }

    console.timeEnd("[EMTAC] render drawings");

    const totalItems =
        (payload.documents?.length || 0) +
        (payload.parts?.length || 0) +
        (payload.images?.length || 0) +
        (payload.drawings?.length || 0);

    if (totalItems === 0) {
        console.log("[EMTAC] Payload loaded but contained no supporting items.");
    }

    console.timeEnd("[EMTAC] payload render total");
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

function limitPayloadArray(items, limit) {
    if (!Array.isArray(items)) {
        return [];
    }

    if (!Number.isFinite(limit) || limit <= 0) {
        return items;
    }

    return items.slice(0, limit);
}

function renderDrawingsPayload(drawings, blocks, meta = {}) {
    const safeDrawings = Array.isArray(drawings) ? drawings : [];
    const totalCount = meta.totalCount || safeDrawings.length;
    const renderedCount = meta.renderedCount || safeDrawings.length;

    const documentsContainer =
        blocks &&
        Array.isArray(blocks["documents-container"])
            ? blocks["documents-container"]
            : [];

    const hasDrawingNavigationBlocks = documentsContainer.some(doc => {
        return (
            doc &&
            doc.drawing_navigation &&
            Array.isArray(doc.drawing_navigation.areas) &&
            doc.drawing_navigation.areas.length > 0
        );
    });

    console.log("[EMTAC] renderDrawingsPayload selected.", {
        drawingsReceived: safeDrawings.length,
        renderedCount: renderedCount,
        totalCount: totalCount,
        hasBlocks: Boolean(blocks),
        documentsContainerCount: documentsContainer.length,
        hasDrawingNavigationBlocks: hasDrawingNavigationBlocks,
        hasBlockRenderer: typeof window.updateDrawingsPanelFromBlocks === "function",
        hasDisplayDrawings: typeof displayDrawings === "function",
        hasRenderDrawings: typeof renderDrawings === "function",
    });

    window.__lastDrawingRenderInputs = {
        drawings: safeDrawings,
        blocks: blocks,
        meta: meta,
        hasDrawingNavigationBlocks: hasDrawingNavigationBlocks,
    };

    if (
        window.EMTAC_FORCE_FLAT_DRAWING_RENDERER !== true &&
        hasDrawingNavigationBlocks &&
        typeof window.updateDrawingsPanelFromBlocks === "function"
    ) {
        console.log("[EMTAC] Rendering drawings from drawing_navigation blocks.");
        window.updateDrawingsPanelFromBlocks(blocks);
        return;
    }

    if (typeof displayDrawings === "function") {
        console.log("[EMTAC] Rendering drawings with displayDrawings().");
        displayDrawings(safeDrawings);
        return;
    }

    if (typeof renderDrawings === "function") {
        console.log("[EMTAC] Rendering drawings with renderDrawings().");
        renderDrawings(safeDrawings);
        return;
    }

    console.log("[EMTAC] Rendering drawings with renderDrawingsSafely().");
    renderDrawingsSafely(safeDrawings);
}

function renderDrawingsSafely(drawings) {
    const safeDrawings = Array.isArray(drawings) ? drawings : [];
    const drawingSection = document.getElementById("drawing-section");

    if (!drawingSection) {
        console.warn("[EMTAC] No drawing renderer found and #drawing-section does not exist.");
        return;
    }

    drawingSection.innerHTML = "";

    if (safeDrawings.length === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No drawings found.";
        drawingSection.appendChild(empty);
        return;
    }

    const fragment = document.createDocumentFragment();

    safeDrawings.forEach((drawing) => {
        const title =
            drawing.title ||
            drawing.drw_name ||
            drawing.name ||
            drawing.drw_number ||
            "Drawing";

        const wrapper = document.createElement("div");
        wrapper.classList.add("drawing-link-wrapper");

        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("drawing-link");
        button.textContent = title;

        button.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();

            if (typeof window.openDrawingDetailsInPage === "function") {
                window.openDrawingDetailsInPage(drawing);
                return;
            }

            if (typeof window.openDrawingDetails === "function") {
                window.openDrawingDetails(drawing);
                return;
            }

            const href =
                drawing.url ||
                drawing.file_url ||
                drawing.file_path ||
                "";

            if (href) {
                window.location.href = href;
            }
        });

        wrapper.appendChild(button);
        fragment.appendChild(wrapper);
    });

    drawingSection.appendChild(fragment);
}

function appendPayloadLimitNotice(containerId, label, renderedCount, totalCount) {
    if (!Number.isFinite(totalCount)) {
        return;
    }

    if (totalCount <= renderedCount) {
        return;
    }

    const container = document.getElementById(containerId);

    if (!container) {
        console.warn("[EMTAC] Cannot append payload limit notice. Missing container:", containerId);
        return;
    }

    const notice = document.createElement("div");
    notice.className = "emtac-payload-limit-notice";
    notice.textContent = `Showing first ${renderedCount} of ${totalCount} ${label}. Full payload is available in window.EMTAC_LAST_PAYLOAD.`;

    container.appendChild(notice);
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

function getActiveDocumentScopeForAsk() {
    let scope = null;

    try {
        if (typeof window.getEMTACActiveDocumentScope === "function") {
            scope = window.getEMTACActiveDocumentScope();
        }

        if (!scope && window.EMTAC_ACTIVE_DOCUMENT_SCOPE) {
            scope = window.EMTAC_ACTIVE_DOCUMENT_SCOPE;
        }

        if (!scope || typeof scope !== "object") {
            return null;
        }

        const completeDocumentId =
            scope.complete_document_id ??
            scope.completed_document_id ??
            scope.completeDocumentId ??
            scope.completeDocumentID ??
            null;

        if (!completeDocumentId) {
            console.warn("[EMTAC DOCUMENT MODE] Active document scope is missing complete_document_id.", scope);
            return null;
        }

        return {
            enabled: scope.enabled !== false,
            scope_type: scope.scope_type || scope.scopeType || "complete_document",
            document_id: scope.document_id ?? scope.documentId ?? null,
            complete_document_id: completeDocumentId,
            document_name:
                scope.document_name ||
                scope.documentName ||
                scope.name ||
                scope.title ||
                "Selected Document",
        };

    } catch (error) {
        console.warn("[EMTAC DOCUMENT MODE] Failed to read active document scope.", error);
        return null;
    }
}

function extractConversationId(data) {
    if (!data || typeof data !== "object") {
        return null;
    }

    const value =
        data.conversation_id ||
        data.conversationId ||
        data.chatSessionId ||
        data.chat_session_id ||
        data.sessionId ||
        data.session_id ||
        null;

    if (value === null || value === undefined) {
        return null;
    }

    const normalized = String(value).trim();

    return normalized || null;
}

function restoreConversationIdFromSessionStorage() {
    try {
        const storedConversationId = sessionStorage.getItem(EMTAC_CONVERSATION_STORAGE_KEY);

        if (!storedConversationId) {
            return null;
        }

        const normalized = String(storedConversationId).trim();

        return normalized || null;

    } catch (error) {
        console.debug("[EMTAC] Unable to restore conversation_id from sessionStorage.", error);
        return null;
    }
}

function persistConversationIdToSessionStorage(conversationId) {
    try {
        if (conversationId) {
            sessionStorage.setItem(EMTAC_CONVERSATION_STORAGE_KEY, conversationId);
            return;
        }

        sessionStorage.removeItem(EMTAC_CONVERSATION_STORAGE_KEY);

    } catch (error) {
        console.debug("[EMTAC] Unable to persist conversation_id to sessionStorage.", error);
    }
}

function syncConversationGlobals(conversationId) {
    window.EMTAC_ACTIVE_CONVERSATION_ID = conversationId || null;
    window.currentConversationId = conversationId || null;
    window.lastConversationId = conversationId || null;
    window.conversationId = conversationId || null;
    window.activeConversationId = conversationId || null;
    window.current_conversation_id = conversationId || null;
}

function getActiveConversationId() {
    if (!isConversationEnabledForAsk()) {
        return null;
    }

    if (activeConversationId) {
        return activeConversationId;
    }

    const restoredConversationId = restoreConversationIdFromSessionStorage();

    if (restoredConversationId) {
        activeConversationId = restoredConversationId;
        syncConversationGlobals(activeConversationId);
        return activeConversationId;
    }

    return null;
}

function setActiveConversationId(conversationId) {
    if (!isConversationEnabledForAsk()) {
        clearActiveConversationId();
        return;
    }

    const normalized = conversationId ? String(conversationId).trim() : null;

    if (!normalized) {
        return;
    }

    activeConversationId = normalized;
    persistConversationIdToSessionStorage(activeConversationId);
    syncConversationGlobals(activeConversationId);

    console.log("[EMTAC] Active conversation_id updated:", activeConversationId);
}

function clearActiveConversationId() {
    activeConversationId = null;
    persistConversationIdToSessionStorage(null);
    syncConversationGlobals(null);

    console.log("[EMTAC] Active conversation_id cleared.");
}

function startNewEMTACChat() {
    clearActiveConversationId();
    activeAnswerRequestId = null;

    try {
        sessionStorage.removeItem("emtac_qanda_feedback_context");
    } catch (error) {
        console.debug("[EMTAC] Unable to clear Q&A feedback context from sessionStorage.", error);
    }

    clearAllContainers();
    setAnswerHtml("Chat cleared. Ask a new question to start a fresh conversation.");
    setPayloadLoadingMessage("");
    setPayloadVisualState("clear");

    const inputEl = document.getElementById("user_input");

    if (inputEl) {
        inputEl.value = "";
        inputEl.focus();
    }

    console.info("[EMTAC] Clear Chat clicked. New conversation will be created on the next question.");
}

// Expose lightweight helpers for debugging from the browser console.
window.getEMTACActiveConversationId = getActiveConversationId;
window.clearEMTACActiveConversationId = clearActiveConversationId;
window.startNewEMTACChat = startNewEMTACChat;
window.getEMTACActiveDocumentScopeForAsk = getActiveDocumentScopeForAsk;

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
            conversation_enabled: isConversationEnabledForAsk(),
            conversation_mode: isConversationEnabledForAsk() ? "conversation" : "single_turn",
            conversation_id: getActiveConversationId(),
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

    if (payloadEndpoint === "/ask/payload") {
        return "/chatbot/ask/payload";
    }

    if (payloadEndpoint === "/chatbot/ask/payload") {
        return payloadEndpoint;
    }

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
    syncConversationModeGlobals(isConversationMemoryEnabled);

    if (!isConversationMemoryEnabled) {
        clearActiveConversationId();
    } else {
        syncConversationGlobals(getActiveConversationId());
    }

    console.log("[EMTAC] Conversation memory initialized:", {
        conversation_enabled: isConversationEnabledForAsk(),
        conversation_mode: isConversationEnabledForAsk() ? "conversation" : "single_turn",
        conversation_id: getActiveConversationId(),
        hasConversation: Boolean(getActiveConversationId()),
    });

    const askBtn = document.getElementById("submit-question");

    if (askBtn && askBtn.dataset.emtacChatbotBound !== "true") {
        askBtn.dataset.emtacChatbotBound = "true";

        askBtn.addEventListener("click", (event) => {
            event.preventDefault();
            submitQuestion(event);
        });
    }

    const inputEl = document.getElementById("user_input");

    if (inputEl && inputEl.dataset.emtacChatbotBound !== "true") {
        inputEl.dataset.emtacChatbotBound = "true";

        inputEl.addEventListener("keypress", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitQuestion(event);
            }
        });
    }

    const clearChatBtn = document.getElementById("clear-chat");

    if (clearChatBtn && clearChatBtn.dataset.emtacChatbotBound !== "true") {
        clearChatBtn.dataset.emtacChatbotBound = "true";

        clearChatBtn.addEventListener("click", (event) => {
            event.preventDefault();
            startNewEMTACChat();
        });
    } else if (!clearChatBtn) {
        console.log("[EMTAC] No #clear-chat button found, skipping clear-chat binding.");
    }

    const toggleBtn = document.getElementById("toggle-voice");

    if (toggleBtn && toggleBtn.dataset.emtacChatbotBound !== "true") {
        toggleBtn.dataset.emtacChatbotBound = "true";
        toggleBtn.addEventListener("click", toggleTextToSpeech);
    } else if (!toggleBtn) {
        console.log("[EMTAC] No #toggle-voice button found, skipping voice binding.");
    }
});