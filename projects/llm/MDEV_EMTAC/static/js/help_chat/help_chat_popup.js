document.addEventListener("DOMContentLoaded", function () {
    console.log("[HELP_CHAT] help_chat_popup.js loaded");

    const helpChatLink = document.getElementById("helpChatSidebarLink");
    const helpChatPopup = document.getElementById("helpChatPopup");
    const helpChatCloseButton = document.getElementById("helpChatCloseButton");
    const helpChatMessages = document.getElementById("helpChatMessages");
    const helpChatInput = document.getElementById("helpChatInput");
    const helpChatSendButton = document.getElementById("helpChatSendButton");

    let socket = null;
    let sessionUuid = null;

    if (!helpChatLink || !helpChatPopup || !helpChatMessages || !helpChatInput || !helpChatSendButton) {
        console.warn("[HELP_CHAT] Missing required help chat elements.");
        return;
    }

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function addMessage(sender, text, senderType) {
        const message = document.createElement("div");
        message.className = "help-chat-message help-chat-message-" + (senderType || "system");

        message.innerHTML = `
            <div class="help-chat-message-sender">${escapeHtml(sender)}</div>
            <div class="help-chat-message-text">${escapeHtml(text)}</div>
        `;

        helpChatMessages.appendChild(message);
        helpChatMessages.scrollTop = helpChatMessages.scrollHeight;
    }

    function renderMessages(messages) {
        helpChatMessages.innerHTML = "";

        if (!messages || messages.length === 0) {
            addMessage("EMTAC Help", "How can I help you?", "system");
            return;
        }

        messages.forEach(function (message) {
            addMessage(
                message.sender_name || message.sender_type || "Unknown",
                message.message_text || "",
                message.sender_type || "system"
            );
        });
    }

    async function ensureSession() {
        const response = await fetch("/help-chat/api/session", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                current_page: window.location.href
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || "Failed to create help chat session.");
        }

        sessionUuid = data.session.session_uuid;
        renderMessages(data.messages || []);

        return data.session;
    }

    function ensureSocket() {
        if (socket || typeof io === "undefined") {
            return;
        }

        socket = io();

        socket.on("connect", function () {
            console.log("[HELP_CHAT] Socket connected");

            if (sessionUuid) {
                socket.emit("help_chat_join_user_room", {
                    session_uuid: sessionUuid
                });
            }
        });

        socket.on("help_chat_new_message", function (message) {
            if (!message) {
                return;
            }

            addMessage(
                message.sender_name || message.sender_type || "Unknown",
                message.message_text || "",
                message.sender_type || "system"
            );
        });

        socket.on("help_chat_error", function (payload) {
            console.warn("[HELP_CHAT] Socket error:", payload);
            addMessage("System", payload.message || "Help chat error.", "system");
        });
    }

    async function openHelpChat() {
        helpChatPopup.style.display = "flex";

        try {
            await ensureSession();
            ensureSocket();

            if (socket && socket.connected && sessionUuid) {
                socket.emit("help_chat_join_user_room", {
                    session_uuid: sessionUuid
                });
            }

            helpChatInput.focus();
        } catch (error) {
            console.error("[HELP_CHAT] Failed to open help chat:", error);
            addMessage("System", error.message || "Failed to open help chat.", "system");
        }
    }

    function closeHelpChat() {
        helpChatPopup.style.display = "none";

        if (socket && sessionUuid) {
            socket.emit("help_chat_leave_user_room", {
                session_uuid: sessionUuid
            });
        }
    }

    function sendHelpMessage() {
        const messageText = helpChatInput.value.trim();

        if (!messageText) {
            return;
        }

        if (!sessionUuid) {
            addMessage("System", "Help chat session is not ready yet.", "system");
            return;
        }

        helpChatInput.value = "";

        if (socket && socket.connected) {
            socket.emit("help_chat_user_send_message", {
                session_uuid: sessionUuid,
                message_text: messageText,
                current_page: window.location.href
            });
            return;
        }

        fetch("/help-chat/api/messages", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message_text: messageText,
                current_page: window.location.href
            })
        })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    throw new Error(data.message || "Failed to send message.");
                }

                addMessage(
                    data.message.sender_name || "You",
                    data.message.message_text || "",
                    data.message.sender_type || "user"
                );
            })
            .catch(error => {
                console.error("[HELP_CHAT] Failed to send message:", error);
                addMessage("System", error.message || "Failed to send message.", "system");
            });
    }

    helpChatLink.addEventListener("click", function (event) {
        event.preventDefault();
        openHelpChat();
    });

    if (helpChatCloseButton) {
        helpChatCloseButton.addEventListener("click", closeHelpChat);
    }

    helpChatSendButton.addEventListener("click", sendHelpMessage);

    helpChatInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendHelpMessage();
        }
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && helpChatPopup.style.display !== "none") {
            closeHelpChat();
        }
    });
});
