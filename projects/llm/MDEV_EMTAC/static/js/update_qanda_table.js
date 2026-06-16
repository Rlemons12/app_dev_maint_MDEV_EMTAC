console.log("[QandAFeedback] update_qanda_table.js loaded - qanda_feedback_20260527_4");

(function () {
    "use strict";

    const FEEDBACK_ENDPOINT = "/chatbot/update_qanda";

    window.EMTAC_QANDA_FEEDBACK_CONTEXT = window.EMTAC_QANDA_FEEDBACK_CONTEXT || {
        userId: null,
        question: "",
        answer: "",
        requestId: null
    };

    window.updateQandAFeedbackContext = function updateQandAFeedbackContext(context) {
        context = context || {};

        window.EMTAC_QANDA_FEEDBACK_CONTEXT = {
            userId: firstUsefulValue(context.userId, context.user_id, "anonymous"),
            question: firstUsefulValue(context.question, ""),
            answer: firstUsefulValue(context.answer, ""),
            requestId: firstUsefulValue(context.requestId, context.request_id, null)
        };

        console.log("[QandAFeedback] Context updated:", window.EMTAC_QANDA_FEEDBACK_CONTEXT);
    };

    document.addEventListener("DOMContentLoaded", function () {
        const submitButton = document.getElementById("submit_comment_rating");

        if (!submitButton) {
            console.log("[QandAFeedback] No #submit_comment_rating button found.");
            return;
        }

        submitButton.addEventListener("click", function (event) {
            event.preventDefault();

            const rating = document.getElementById("rating")?.value || "";
            const comment = document.getElementById("comment")?.value?.trim() || "";

            updateQandATable(rating, comment);
        });
    });

    async function updateQandATable(rating, comment) {
        const context = window.EMTAC_QANDA_FEEDBACK_CONTEXT || {};

        const originalRequestId = firstUsefulValue(
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
            request_id: originalRequestId,
            requestId: originalRequestId
        };

        console.log("[QandAFeedback] Submitting payload:", payload);

        if (!payload.question || !payload.answer || !payload.request_id) {
            console.error(
                "[QandAFeedback] Missing Q&A context. Ask a question before submitting feedback.",
                payload
            );
            alert("Ask a question first, then submit feedback.");
            return;
        }

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

            const result = await response.json().catch(function () {
                return null;
            });

            console.log("[QandAFeedback] Response:", {
                ok: response.ok,
                status: response.status,
                result: result
            });

            if (!response.ok || !result || result.status !== "success") {
                console.error("[QandAFeedback] Failed to update Q&A table.", result);
                alert(result?.message || "Failed to save feedback.");
                return;
            }

            console.log("[QandAFeedback] Q&A table updated successfully.");

            const commentElement = document.getElementById("comment");
            const ratingElement = document.getElementById("rating");

            if (commentElement) {
                commentElement.value = "";
            }

            if (ratingElement) {
                ratingElement.value = "1";
            }

            alert("Feedback saved.");

        } catch (error) {
            console.error("[QandAFeedback] Error submitting feedback:", error);
            alert("Error submitting feedback. Check browser console and Flask logs.");
        }
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