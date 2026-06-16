document.addEventListener("DOMContentLoaded", function () {
    console.log("[HELP_CHAT_ADMIN] help_chat_admin.js loaded");

    const adminLink = document.getElementById("helpChatAdminSidebarLink");
    const adminPopup = document.getElementById("helpChatAdminPopup");
    const adminCloseButton = document.getElementById("helpChatAdminCloseButton");

    const tabsContainer = document.getElementById("helpChatAdminSessionTabs");
    const messagesContainer = document.getElementById("helpChatAdminMessages");
    const activeHeader = document.getElementById("helpChatAdminActiveHeader");
    const input = document.getElementById("helpChatAdminInput");
    const sendButton = document.getElementById("helpChatAdminSendButton");
    const refreshButton = document.getElementById("helpChatRefreshSessionsButton");

    let socket = null;
    let activeSessionUuid = null;
    let sessions = [];

    if (!adminLink || !adminPopup) {
        console.warn("[HELP_CHAT_ADMIN] Admin popup link or popup not found.");
        return;
    }

    if (!tabsContainer || !messagesContainer || !activeHeader || !input || !sendButton || !refreshButton) {
        console.warn("[HELP_CHAT_ADMIN] One or more admin chat elements are missing.");
        return;
    }

    function startHelpChatAdminPulse() {
        if (!adminPopup.classList.contains("is-open")) {
            adminLink.classList.add("has-alert");
        }
    }

    function stopHelpChatAdminPulse() {
        adminLink.classList.remove("has-alert");
    }

    function sessionHasWaitingQuestion(chatSession) {
        if (!chatSession) {
            return false;
        }

        const isOnline = Boolean(chatSession.is_online);

        const hasUserMessage =
            chatSession.last_sender_type === "user" ||
            chatSession.last_message_sender_type === "user" ||
            chatSession.has_waiting_question === true ||
            chatSession.waiting_questions > 0 ||
            chatSession.unread_count > 0;

        return isOnline && hasUserMessage;
    }

    function updateAdminPulseFromSessions() {
        const shouldPulse = sessions.some(sessionHasWaitingQuestion);

        if (shouldPulse) {
            startHelpChatAdminPulse();
        } else {
            stopHelpChatAdminPulse();
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

    function formatSessionLabel(chatSession) {
        const name =
            chatSession.display_name ||
            chatSession.employee_id ||
            chatSession.user_id ||
            "Unknown User";

        const status = chatSession.is_online ? "online" : "offline";

        return `${name} (${status})`;
    }

    function setInputEnabled(enabled) {
        input.disabled = !enabled;
        sendButton.disabled = !enabled;
    }

    function renderTabs() {
        tabsContainer.innerHTML = "";

        if (!sessions.length) {
            tabsContainer.innerHTML = '<div class="help-chat-admin-empty">No help chats yet.</div>';
            return;
        }

        sessions.forEach(function (chatSession) {
            const tab = document.createElement("button");
            tab.type = "button";
            tab.className = "help-chat-admin-tab";

            if (chatSession.session_uuid === activeSessionUuid) {
                tab.classList.add("active");
            }

            if (sessionHasWaitingQuestion(chatSession)) {
                tab.classList.add("has-waiting-question");
            }

            tab.innerHTML = `
                <div class="help-chat-admin-tab-name">${escapeHtml(formatSessionLabel(chatSession))}</div>
                <div class="help-chat-admin-tab-meta">${escapeHtml(chatSession.current_page || "")}</div>
                <div class="help-chat-admin-tab-preview">${escapeHtml(chatSession.last_message_preview || "")}</div>
            `;

            tab.addEventListener("click", function () {
                loadSessionMessages(chatSession.session_uuid).catch(function (error) {
                    console.error("[HELP_CHAT_ADMIN] Failed to load selected session:", error);
                });
            });

            tabsContainer.appendChild(tab);
        });
    }

    function addMessage(message) {
        const senderType = message.sender_type || "system";

        const item = document.createElement("div");
        item.className = "help-chat-admin-message help-chat-admin-message-" + senderType;

        item.innerHTML = `
            <div class="help-chat-admin-message-meta">
                <strong>${escapeHtml(message.sender_name || senderType || "Unknown")}</strong>
                <span>${escapeHtml(message.created_at || "")}</span>
            </div>
            <div class="help-chat-admin-message-text">${escapeHtml(message.message_text || "")}</div>
        `;

        messagesContainer.appendChild(item);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function renderMessages(messages) {
        messagesContainer.innerHTML = "";

        if (!messages || !messages.length) {
            messagesContainer.innerHTML = '<div class="help-chat-admin-empty">No messages in this chat yet.</div>';
            return;
        }

        messages.forEach(addMessage);
    }

    async function loadSessions() {
        tabsContainer.innerHTML = '<div class="help-chat-admin-empty">Loading chats...</div>';

        const response = await fetch("/help-chat/admin/api/sessions", {
            method: "GET",
            credentials: "same-origin"
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Failed to load sessions.");
        }

        sessions = data.sessions || [];
        renderTabs();
        updateAdminPulseFromSessions();

        if (!activeSessionUuid && sessions.length) {
            await loadSessionMessages(sessions[0].session_uuid);
        }

        if (!sessions.length) {
            activeSessionUuid = null;
            activeHeader.innerHTML = "Select a chat.";
            messagesContainer.innerHTML = '<div class="help-chat-admin-empty">No help chats yet.</div>';
            setInputEnabled(false);
        }
    }

    async function loadSessionMessages(sessionUuid) {
        if (!sessionUuid) {
            return;
        }

        activeSessionUuid = sessionUuid;
        renderTabs();

        const response = await fetch(
            `/help-chat/admin/api/session/${encodeURIComponent(sessionUuid)}/messages`,
            {
                method: "GET",
                credentials: "same-origin"
            }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Failed to load messages.");
        }

        activeHeader.innerHTML = `
            <strong>${escapeHtml(data.session.display_name || "Unknown User")}</strong>
            <span>${escapeHtml(data.session.current_page || "")}</span>
        `;

        setInputEnabled(true);
        renderMessages(data.messages || []);
    }

    function ensureSocket() {
        if (socket) {
            if (socket.connected) {
                socket.emit("help_chat_admin_join");
            }
            return;
        }

        if (typeof io === "undefined") {
            console.warn("[HELP_CHAT_ADMIN] Socket.IO client is not loaded.");
            return;
        }

        socket = io();

        socket.on("connect", function () {
            console.log("[HELP_CHAT_ADMIN] Socket connected");
            socket.emit("help_chat_admin_join");
        });

        socket.on("disconnect", function () {
            console.warn("[HELP_CHAT_ADMIN] Socket disconnected");
        });

        socket.on("help_chat_admin_new_message", function () {
            loadSessions().catch(console.error);

            if (activeSessionUuid) {
                loadSessionMessages(activeSessionUuid).catch(console.error);
            }
        });

        socket.on("help_chat_new_message", function () {
            startHelpChatAdminPulse();

            if (activeSessionUuid) {
                loadSessionMessages(activeSessionUuid).catch(console.error);
            }

            loadSessions().catch(console.error);
        });

        socket.on("help_chat_admin_message_sent", function () {
            if (activeSessionUuid) {
                loadSessionMessages(activeSessionUuid).catch(console.error);
            }

            loadSessions().catch(console.error);
        });

        socket.on("help_chat_session_updated", function () {
            loadSessions().catch(console.error);
        });

        socket.on("help_chat_error", function (payload) {
            console.warn("[HELP_CHAT_ADMIN] Socket error:", payload);
            alert((payload && payload.message) || "Help chat admin error.");
        });
    }

    function sendAdminMessage() {
        const messageText = input.value.trim();

        if (!messageText || !activeSessionUuid) {
            return;
        }

        if (!socket || !socket.connected) {
            alert("Admin chat socket is not connected yet.");
            return;
        }

        input.value = "";

        socket.emit("help_chat_admin_send_message", {
            session_uuid: activeSessionUuid,
            message_text: messageText
        });
    }

    function openAdminPopup() {
        stopHelpChatAdminPulse();

        adminPopup.classList.add("is-open");
        ensureSocket();

        loadSessions().catch(function (error) {
            console.error("[HELP_CHAT_ADMIN] Failed to load sessions:", error);
        });
    }

    function closeAdminPopup() {
        adminPopup.classList.remove("is-open");
        updateAdminPulseFromSessions();
    }

    adminLink.addEventListener("click", function (event) {
        event.preventDefault();
        openAdminPopup();
    });

    if (adminCloseButton) {
        adminCloseButton.addEventListener("click", closeAdminPopup);
    }

    sendButton.addEventListener("click", sendAdminMessage);

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendAdminMessage();
        }
    });

    refreshButton.addEventListener("click", function () {
        loadSessions().catch(function (error) {
            console.error("[HELP_CHAT_ADMIN] Failed to refresh sessions:", error);
        });
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && adminPopup.classList.contains("is-open")) {
            closeAdminPopup();
        }
    });

    ensureSocket();

    loadSessions().catch(function (error) {
        console.error("[HELP_CHAT_ADMIN] Initial session load failed:", error);
    });
});