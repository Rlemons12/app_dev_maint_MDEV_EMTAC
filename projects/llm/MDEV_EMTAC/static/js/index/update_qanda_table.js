/* static/js/update_qanda_table.js */
console.log("[QandAFeedback] update_qanda_table.js loaded - qanda_feedback_20260527_6_CONVERSATION_ID_FIX");

(function () {
    "use strict";

    const FEEDBACK_ENDPOINT = "/chatbot/update_qanda";
    const FEEDBACK_CONTEXT_STORAGE_KEY = "emtac_qanda_feedback_context";
    const CONVERSATION_STORAGE_KEY = "emtac_active_conversation_id";

    /**
     * Shared feedback context.
     *
     * chatbot.js should update this after each successful chatbot answer.
     * This lets the rating/comment form know which Q&A record it belongs to.
     *
     * Important:
     * Keep both conversationId and conversation_id.
     * Some frontend code uses camelCase.
     * Flask route/backend expects snake_case.
     */
    window.EMTAC_QANDA_FEEDBACK_CONTEXT = normalizeFeedbackContext(
        window.EMTAC_QANDA_FEEDBACK_CONTEXT || {
            userId: null,
            user_id: null,
            question: "",
            answer: "",
            requestId: null,
            request_id: null,
            conversationId: null,
            conversation_id: null
        }
    );

    /**
     * Public helper.
     *
     * Call this from chatbot.js after /chatbot/ask returns.
     *
     * Example:
     * window.updateQandAFeedbackContext({
     *     userId: userId,
     *     question: question,
     *     answer: result.answer,
     *     requestId: result.request_id,
     *     conversationId: result.conversation_id,
     *     conversation_id: result.conversation_id
     * });
     */
    window.updateQandAFeedbackContext = function updateQandAFeedbackContext(context) {
        if (!context || typeof context !== "object") {
            console.debug("[QandAFeedback] updateQandAFeedbackContext ignored invalid context:", context);
            return;
        }

        const existing = normalizeFeedbackContext(window.EMTAC_QANDA_FEEDBACK_CONTEXT || {});
        const resolvedConversationId = resolveConversationId(context, existing);
        const resolvedRequestId = firstUsefulValue(
            context.requestId,
            context.request_id,
            context.requestID,
            existing.requestId,
            existing.request_id,
            window.currentRequestId,
            window.lastRequestId,
            getValueById("request_id"),
            getValueById("requestId")
        );

        const resolvedUserId = firstUsefulValue(
            context.userId,
            context.user_id,
            existing.userId,
            existing.user_id,
            window.currentUserId,
            window.userId,
            getValueById("user_id"),
            getValueById("userId"),
            "anonymous"
        );

        const resolvedQuestion = firstUsefulValue(
            context.question,
            existing.question,
            window.currentQuestion,
            window.lastQuestion,
            getValueById("question"),
            getTextById("current_question"),
            getTextById("last_question")
        );

        const resolvedAnswer = firstUsefulValue(
            context.answer,
            existing.answer,
            window.currentAnswer,
            window.lastAnswer,
            getTextById("current_answer"),
            getTextById("last_answer"),
            getTextById("chatbot-answer"),
            getTextById("answer")
        );

        window.EMTAC_QANDA_FEEDBACK_CONTEXT = normalizeFeedbackContext({
            userId: resolvedUserId,
            user_id: resolvedUserId,

            question: resolvedQuestion,
            answer: resolvedAnswer,

            requestId: resolvedRequestId,
            request_id: resolvedRequestId,

            conversationId: resolvedConversationId,
            conversation_id: resolvedConversationId
        });

        persistFeedbackContextToSessionStorage(window.EMTAC_QANDA_FEEDBACK_CONTEXT);

        console.debug("[QandAFeedback] conversation_id resolved:", {
            contextConversationId: context.conversationId,
            contextConversation_id: context.conversation_id,
            existingConversationId: existing.conversationId,
            existingConversation_id: existing.conversation_id,
            windowCurrentConversationId: window.currentConversationId,
            windowLastConversationId: window.lastConversationId,
            windowEMTACConversationId: window.EMTAC_ACTIVE_CONVERSATION_ID,
            storedConversationId: getStoredConversationId(),
            finalConversationId: window.EMTAC_QANDA_FEEDBACK_CONTEXT.conversationId,
            finalConversation_id: window.EMTAC_QANDA_FEEDBACK_CONTEXT.conversation_id
        });

        console.debug("[QandAFeedback] Context updated:", window.EMTAC_QANDA_FEEDBACK_CONTEXT);
    };

    document.addEventListener("DOMContentLoaded", function () {
        restoreFeedbackContextFromSessionStorage();

        const submitButton = document.getElementById("submit_comment_rating");

        if (!submitButton) {
            console.debug("[QandAFeedback] submit_comment_rating button not found on this page.");
            return;
        }

        if (submitButton.dataset.qandaFeedbackBound === "true") {
            console.debug("[QandAFeedback] submit_comment_rating already bound. Skipping duplicate binding.");
            return;
        }

        submitButton.dataset.qandaFeedbackBound = "true";

        submitButton.addEventListener("click", function (event) {
            event.preventDefault();

            const ratingElement = document.getElementById("rating");
            const commentElement = document.getElementById("comment");

            const rating = ratingElement ? ratingElement.value : "";
            const comment = commentElement ? commentElement.value.trim() : "";

            submitQandAFeedback({
                rating: rating,
                comment: comment,
                ratingElement: ratingElement,
                commentElement: commentElement,
                submitButton: submitButton
            });
        });
    });

    document.addEventListener("emtac:qanda-context", function (event) {
        if (event && event.detail) {
            window.updateQandAFeedbackContext(event.detail);
        }
    });

    document.addEventListener("emtac:conversation-cleared", function () {
        clearFeedbackConversationContext();
    });

    async function submitQandAFeedback(options) {
        const rating = options.rating;
        const comment = options.comment;
        const ratingElement = options.ratingElement;
        const commentElement = options.commentElement;
        const submitButton = options.submitButton;

        if (!comment && !rating) {
            showFeedbackMessage("Please select a rating or enter a comment.", "warning");
            return;
        }

        const context = getFeedbackContext();

        const payloadConversationId = firstUsefulValue(
            context.conversationId,
            context.conversation_id,
            getStoredConversationId(),
            null
        );

        const payloadRequestId = firstUsefulValue(
            context.requestId,
            context.request_id,
            null
        );

        const payload = {
            user_id: firstUsefulValue(context.userId, context.user_id, "anonymous"),
            question: firstUsefulValue(context.question, ""),
            answer: firstUsefulValue(context.answer, ""),
            rating: rating,
            comment: comment,

            // Original /chatbot/ask request_id. Backend uses this to find the QandA row.
            request_id: payloadRequestId,
            requestId: payloadRequestId,

            // Active ChatSession conversation ID. Backend logs/uses this for memory-aware feedback.
            conversation_id: payloadConversationId,
            conversationId: payloadConversationId
        };

        console.debug("[QandAFeedback] Submit payload prepared:", {
            user_id: payload.user_id,
            hasQuestion: Boolean(payload.question),
            hasAnswer: Boolean(payload.answer),
            rating: payload.rating,
            hasComment: Boolean(payload.comment),
            request_id: payload.request_id,
            conversation_id: payload.conversation_id
        });

        if (!payload.question || !payload.answer) {
            console.warn(
                "[QandAFeedback] Missing question or answer. Backend may not know which Q&A row to update.",
                payload
            );

            showFeedbackMessage(
                "Feedback could not be linked to the latest answer. Ask a question first, then submit feedback.",
                "error"
            );

            return;
        }

        if (!payload.request_id) {
            console.warn(
                "[QandAFeedback] Missing request_id. Feedback may not match the exact Q&A row.",
                payload
            );
        }

        if (!payload.conversation_id) {
            console.warn(
                "[QandAFeedback] Missing conversation_id. Feedback will still save by request_id, but it will not be tied to the active memory thread.",
                payload
            );
        }

        setSubmitState(submitButton, true);

        try {
            const response = await fetch(FEEDBACK_ENDPOINT, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                credentials: "same-origin",
                body: JSON.stringify(payload)
            });

            const result = await safeReadJson(response);

            if (!response.ok || !result || result.status !== "success") {
                const message =
                    result && result.message
                        ? result.message
                        : "Failed to update Q&A feedback.";

                console.error("[QandAFeedback] Feedback save failed:", {
                    status: response.status,
                    result: result
                });

                showFeedbackMessage(message, "error");
                return;
            }

            console.info("[QandAFeedback] Q&A table updated successfully.", result);

            if (commentElement) {
                commentElement.value = "";
            }

            if (ratingElement) {
                ratingElement.value = "1";
            }

            showFeedbackMessage("Feedback saved.", "success");

        } catch (error) {
            console.error("[QandAFeedback] Error submitting feedback:", error);
            showFeedbackMessage("Error submitting feedback. Check the console/logs.", "error");

        } finally {
            setSubmitState(submitButton, false);
        }
    }

    function getFeedbackContext() {
        const existing = normalizeFeedbackContext(window.EMTAC_QANDA_FEEDBACK_CONTEXT || {});
        const resolvedConversationId = resolveConversationId({}, existing);
        const resolvedRequestId = firstUsefulValue(
            existing.requestId,
            existing.request_id,
            window.currentRequestId,
            window.lastRequestId,
            getValueById("request_id"),
            getValueById("requestId")
        );
        const resolvedUserId = firstUsefulValue(
            existing.userId,
            existing.user_id,
            window.currentUserId,
            window.userId,
            getValueById("user_id"),
            getValueById("userId"),
            "anonymous"
        );

        const context = normalizeFeedbackContext({
            userId: resolvedUserId,
            user_id: resolvedUserId,

            question: firstUsefulValue(
                existing.question,
                window.currentQuestion,
                window.lastQuestion,
                getValueById("question"),
                getTextById("current_question"),
                getTextById("last_question")
            ),

            answer: firstUsefulValue(
                existing.answer,
                window.currentAnswer,
                window.lastAnswer,
                getTextById("current_answer"),
                getTextById("last_answer"),
                getTextById("chatbot-answer"),
                getTextById("answer")
            ),

            requestId: resolvedRequestId,
            request_id: resolvedRequestId,

            conversationId: resolvedConversationId,
            conversation_id: resolvedConversationId
        });

        window.EMTAC_QANDA_FEEDBACK_CONTEXT = context;
        persistFeedbackContextToSessionStorage(context);

        return context;
    }

    function restoreFeedbackContextFromSessionStorage() {
        try {
            const rawContext = sessionStorage.getItem(FEEDBACK_CONTEXT_STORAGE_KEY);

            if (!rawContext) {
                window.EMTAC_QANDA_FEEDBACK_CONTEXT = normalizeFeedbackContext(
                    window.EMTAC_QANDA_FEEDBACK_CONTEXT || {}
                );
                return;
            }

            const parsedContext = JSON.parse(rawContext);

            if (parsedContext && typeof parsedContext === "object") {
                window.EMTAC_QANDA_FEEDBACK_CONTEXT = normalizeFeedbackContext({
                    ...window.EMTAC_QANDA_FEEDBACK_CONTEXT,
                    ...parsedContext
                });

                const resolvedConversationId = resolveConversationId(
                    {},
                    window.EMTAC_QANDA_FEEDBACK_CONTEXT
                );

                if (resolvedConversationId) {
                    window.EMTAC_QANDA_FEEDBACK_CONTEXT.conversationId = resolvedConversationId;
                    window.EMTAC_QANDA_FEEDBACK_CONTEXT.conversation_id = resolvedConversationId;
                }

                persistFeedbackContextToSessionStorage(window.EMTAC_QANDA_FEEDBACK_CONTEXT);

                console.debug("[QandAFeedback] Restored feedback context:", window.EMTAC_QANDA_FEEDBACK_CONTEXT);
            }
        } catch (error) {
            console.debug("[QandAFeedback] Unable to restore feedback context from sessionStorage.", error);

            window.EMTAC_QANDA_FEEDBACK_CONTEXT = normalizeFeedbackContext(
                window.EMTAC_QANDA_FEEDBACK_CONTEXT || {}
            );
        }
    }

    function clearFeedbackConversationContext() {
        const existing = normalizeFeedbackContext(window.EMTAC_QANDA_FEEDBACK_CONTEXT || {});

        existing.conversationId = null;
        existing.conversation_id = null;
        existing.requestId = null;
        existing.request_id = null;

        window.EMTAC_QANDA_FEEDBACK_CONTEXT = existing;
        persistFeedbackContextToSessionStorage(existing);

        console.debug("[QandAFeedback] Feedback conversation context cleared.");
    }

    function normalizeFeedbackContext(context) {
        const safeContext = context && typeof context === "object"
            ? context
            : {};

        const resolvedUserId = firstUsefulValue(
            safeContext.userId,
            safeContext.user_id,
            null
        );

        const resolvedRequestId = firstUsefulValue(
            safeContext.requestId,
            safeContext.request_id,
            null
        );

        const resolvedConversationId = firstUsefulValue(
            safeContext.conversationId,
            safeContext.conversation_id,
            null
        );

        return {
            userId: resolvedUserId,
            user_id: resolvedUserId,

            question: firstUsefulValue(safeContext.question, ""),
            answer: firstUsefulValue(safeContext.answer, ""),

            requestId: resolvedRequestId,
            request_id: resolvedRequestId,

            conversationId: resolvedConversationId,
            conversation_id: resolvedConversationId
        };
    }

    function resolveConversationId(context, existing) {
        const incoming = context && typeof context === "object" ? context : {};
        const current = existing && typeof existing === "object" ? existing : {};

        return firstUsefulValue(
            incoming.conversationId,
            incoming.conversation_id,
            incoming.chatSessionId,
            incoming.chat_session_id,
            incoming.sessionId,
            incoming.session_id,

            current.conversationId,
            current.conversation_id,
            current.chatSessionId,
            current.chat_session_id,
            current.sessionId,
            current.session_id,

            window.currentConversationId,
            window.lastConversationId,
            window.EMTAC_ACTIVE_CONVERSATION_ID,

            getValueById("conversation_id"),
            getValueById("conversationId"),
            getValueById("chatSessionId"),
            getValueById("chat_session_id"),

            getStoredConversationId()
        );
    }

    function getStoredConversationId() {
        try {
            const value = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);

            if (!value) {
                return "";
            }

            return String(value).trim();

        } catch (error) {
            console.debug("[QandAFeedback] Unable to read conversation_id from sessionStorage.", error);
            return "";
        }
    }

    function persistFeedbackContextToSessionStorage(context) {
        try {
            sessionStorage.setItem(
                FEEDBACK_CONTEXT_STORAGE_KEY,
                JSON.stringify(normalizeFeedbackContext(context))
            );
        } catch (error) {
            console.debug("[QandAFeedback] Unable to save feedback context to sessionStorage.", error);
        }
    }

    function safeReadJson(response) {
        return response
            .json()
            .catch(function () {
                return null;
            });
    }

    function setSubmitState(button, isSubmitting) {
        if (!button) {
            return;
        }

        button.disabled = isSubmitting;
        button.dataset.originalText = button.dataset.originalText || button.textContent;
        button.textContent = isSubmitting ? "Saving..." : button.dataset.originalText;
    }

    function showFeedbackMessage(message, level) {
        const normalizedLevel = level || "info";

        let messageBox = document.getElementById("qanda-feedback-message");

        if (!messageBox) {
            messageBox = document.createElement("div");
            messageBox.id = "qanda-feedback-message";
            messageBox.setAttribute("role", "status");
            messageBox.style.marginTop = "0.75rem";
            messageBox.style.padding = "0.5rem 0.75rem";
            messageBox.style.borderRadius = "0.25rem";
            messageBox.style.fontSize = "0.95rem";

            const submitButton = document.getElementById("submit_comment_rating");

            if (submitButton && submitButton.parentNode) {
                submitButton.parentNode.insertBefore(messageBox, submitButton.nextSibling);
            } else {
                document.body.appendChild(messageBox);
            }
        }

        messageBox.textContent = message;

        if (normalizedLevel === "success") {
            messageBox.style.background = "#d4edda";
            messageBox.style.color = "#155724";
            messageBox.style.border = "1px solid #c3e6cb";
        } else if (normalizedLevel === "warning") {
            messageBox.style.background = "#fff3cd";
            messageBox.style.color = "#856404";
            messageBox.style.border = "1px solid #ffeeba";
        } else if (normalizedLevel === "error") {
            messageBox.style.background = "#f8d7da";
            messageBox.style.color = "#721c24";
            messageBox.style.border = "1px solid #f5c6cb";
        } else {
            messageBox.style.background = "#d1ecf1";
            messageBox.style.color = "#0c5460";
            messageBox.style.border = "1px solid #bee5eb";
        }
    }

    function getValueById(id) {
        const element = document.getElementById(id);

        if (!element) {
            return "";
        }

        if (typeof element.value === "string") {
            return element.value.trim();
        }

        return "";
    }

    function getTextById(id) {
        const element = document.getElementById(id);

        if (!element) {
            return "";
        }

        return (element.textContent || "").trim();
    }

    function firstUsefulValue() {
        for (let index = 0; index < arguments.length; index += 1) {
            const value = arguments[index];

            if (value === null || value === undefined) {
                continue;
            }

            if (typeof value === "string" && value.trim() === "") {
                continue;
            }

            return value;
        }

        return "";
    }
})();