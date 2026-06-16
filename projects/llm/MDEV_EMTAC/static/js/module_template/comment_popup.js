/* static/js/module_template/comment_popup.js
=========================================================
COMMENT POPUP CONTROLLER
Single owner for popup open / close / submit / overlay / escape.
========================================================= */

console.log("[COMMENT POPUP] comment_popup.js loaded");

(function () {
    function getPopupRootFromElement(element) {
        return element ? element.closest('[data-comment-popup-root="true"]') : null;
    }

    function getPrimaryPopupRoot() {
        return document.querySelector('[data-comment-popup-root="true"]');
    }

    function getPopupParts(root) {
        if (!root) {
            return {
                root: null,
                overlay: null,
                popup: null,
                form: null,
                textarea: null,
                imageDataInput: null,
                fileUpload: null,
                pastedImagePreview: null,
                pastedImageFeedback: null
            };
        }

        return {
            root: root,
            overlay: root.querySelector("#commentPopupOverlay"),
            popup: root.querySelector("#commentPopup"),
            form: root.querySelector("#commentForm"),
            textarea: root.querySelector("#comment"),
            imageDataInput: root.querySelector("#imageData"),
            fileUpload: root.querySelector("#fileUpload"),
            pastedImagePreview: root.querySelector("#pastedImagePreview"),
            pastedImageFeedback: root.querySelector("#pastedImageFeedback")
        };
    }

    function showElement(element) {
        if (element) {
            element.style.display = "block";
        }
    }

    function hideElement(element) {
        if (element) {
            element.style.display = "none";
        }
    }

    function resetPreview(parts) {
        if (parts.pastedImagePreview) {
            parts.pastedImagePreview.src = "";
            hideElement(parts.pastedImagePreview);
        }

        if (parts.pastedImageFeedback) {
            hideElement(parts.pastedImageFeedback);
        }

        if (parts.imageDataInput) {
            parts.imageDataInput.value = "";
        }

        if (parts.fileUpload) {
            parts.fileUpload.value = "";
        }
    }

    function openPopup(root, event) {
        const parts = getPopupParts(root);

        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }

        if (!parts.overlay) {
            console.warn("[COMMENT POPUP] Overlay not found.");
            return;
        }

        parts.overlay.hidden = false;
        parts.overlay.setAttribute("aria-hidden", "false");
        document.body.classList.add("comment-popup-open");

        if (parts.textarea) {
            window.setTimeout(function () {
                parts.textarea.focus();
            }, 0);
        }

        console.log("[COMMENT POPUP] Opened");
    }

    function closePopup(root, event) {
        const parts = getPopupParts(root);

        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }

        if (!parts.overlay) {
            console.warn("[COMMENT POPUP] Overlay not found for close.");
            return;
        }

        parts.overlay.hidden = true;
        parts.overlay.setAttribute("aria-hidden", "true");
        document.body.classList.remove("comment-popup-open");

        console.log("[COMMENT POPUP] Closed");
    }

    function initializePopupState() {
        const roots = document.querySelectorAll('[data-comment-popup-root="true"]');

        if (!roots.length) {
            console.warn("[COMMENT POPUP] No popup roots found.");
            return;
        }

        roots.forEach(function (root) {
            const parts = getPopupParts(root);

            if (parts.overlay) {
                parts.overlay.hidden = true;
                parts.overlay.setAttribute("aria-hidden", "true");
            }
        });

        document.body.classList.remove("comment-popup-open");

        console.log("[COMMENT POPUP] Initial state applied to", roots.length, "popup root(s)");
    }

    function bindPasteAndUploadHandlers(root) {
        const parts = getPopupParts(root);

        if (!parts.root || parts.root.dataset.commentPopupMediaBound === "true") {
            return;
        }

        parts.root.dataset.commentPopupMediaBound = "true";

        if (parts.textarea) {
            parts.textarea.addEventListener("paste", function (event) {
                const clipboardItems = event.clipboardData ? event.clipboardData.items : [];

                for (let i = 0; i < clipboardItems.length; i += 1) {
                    const item = clipboardItems[i];

                    if (item.type && item.type.indexOf("image") !== -1) {
                        const file = item.getAsFile();

                        if (!file) {
                            continue;
                        }

                        const reader = new FileReader();
                        reader.onload = function (loadEvent) {
                            if (parts.pastedImagePreview) {
                                parts.pastedImagePreview.src = loadEvent.target.result;
                                showElement(parts.pastedImagePreview);
                            }

                            if (parts.pastedImageFeedback) {
                                showElement(parts.pastedImageFeedback);
                            }

                            if (parts.imageDataInput) {
                                parts.imageDataInput.value = loadEvent.target.result;
                            }
                        };
                        reader.readAsDataURL(file);
                        break;
                    }
                }
            });
        }

        if (parts.fileUpload) {
            parts.fileUpload.addEventListener("change", function () {
                const file = parts.fileUpload.files && parts.fileUpload.files[0];

                if (!file) {
                    resetPreview(parts);
                    return;
                }

                const reader = new FileReader();
                reader.onload = function (event) {
                    if (parts.pastedImagePreview) {
                        parts.pastedImagePreview.src = event.target.result;
                        showElement(parts.pastedImagePreview);
                    }

                    if (parts.pastedImageFeedback) {
                        showElement(parts.pastedImageFeedback);
                    }

                    if (parts.imageDataInput) {
                        parts.imageDataInput.value = event.target.result;
                    }
                };
                reader.readAsDataURL(file);
            });
        }

        if (parts.form) {
            parts.form.addEventListener("submit", function (event) {
                event.preventDefault();
                console.log("[COMMENT POPUP] Submit intercepted");
                closePopup(root);
            });
        }
    }

    function initCommentPopup() {
        initializePopupState();

        const roots = document.querySelectorAll('[data-comment-popup-root="true"]');
        roots.forEach(function (root) {
            bindPasteAndUploadHandlers(root);
        });

        document.addEventListener("click", function (event) {
            const openLink = event.target.closest("#commentPopupLink");
            if (openLink) {
                const root = getPrimaryPopupRoot();
                if (!root) {
                    console.warn("[COMMENT POPUP] No popup root found for opener.");
                    return;
                }

                openPopup(root, event);
                return;
            }

            const closeButton = event.target.closest("#closePopup, #cancelPopup");
            if (closeButton) {
                const root = getPopupRootFromElement(closeButton);
                if (!root) {
                    console.warn("[COMMENT POPUP] No popup root found for close button.");
                    return;
                }

                console.log("[COMMENT POPUP] Close button clicked:", closeButton.id);
                closePopup(root, event);
                return;
            }

            const overlay = event.target.closest(".comment-popup-overlay");
            if (overlay && event.target === overlay) {
                const root = getPopupRootFromElement(overlay);
                if (!root) {
                    console.warn("[COMMENT POPUP] No popup root found for overlay click.");
                    return;
                }

                console.log("[COMMENT POPUP] Overlay clicked");
                closePopup(root, event);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key !== "Escape") {
                return;
            }

            const openOverlay = document.querySelector('.comment-popup-overlay[aria-hidden="false"]');

            if (!openOverlay) {
                return;
            }

            const root = getPopupRootFromElement(openOverlay);
            if (!root) {
                return;
            }

            console.log("[COMMENT POPUP] Escape pressed");
            closePopup(root, event);
        });

        console.log("[COMMENT POPUP] Popup initialized");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initCommentPopup, { once: true });
    } else {
        initCommentPopup();
    }
})();