/* static/js/module_template/top_header_with_comment_popup.js
==================================
TOP HEADER WITH COMMENT POPUP
Single-owner behavior for:
- banner height sync
- popup open / close
- AJAX submit handling
- overlay click
- escape key
- paste / upload preview
==================================
*/

console.log("[THCP] top_header_with_comment_popup.js loaded");

(function () {
    function initTHCP() {
        const root = document.querySelector('[data-thcp-root="true"]');

        if (!root) {
            console.warn("[THCP] Root not found");
            return;
        }

        if (root.dataset.thcpInitialized === "true") {
            console.log("[THCP] Already initialized, skipping duplicate bind.");
            return;
        }

        root.dataset.thcpInitialized = "true";

        const banner = root.querySelector(".thcp-banner");
        const openLink = root.querySelector("#thcpCommentOpen");
        const overlay = root.querySelector("#thcpCommentOverlay");
        const dialog = root.querySelector("#thcpCommentDialog");
        const closeX = root.querySelector("#thcpCommentCloseX");
        const closeBtn = root.querySelector("#thcpCommentCloseBtn");
        const form = root.querySelector("#thcpCommentForm");
        const textarea = root.querySelector("#thcpCommentText");
        const pageUrlInput = root.querySelector("#thcpPageUrl");
        const submitUrlInput = root.querySelector("#thcpSubmitUrl");
        const imageDataInput = root.querySelector("#thcpImageData");
        const fileUpload = root.querySelector("#thcpFileUpload");
        const imagePreview = root.querySelector("#thcpImagePreview");
        const imageFeedback = root.querySelector("#thcpImageFeedback");
        const sendButton = root.querySelector("#thcpCommentSend");

        if (!banner || !openLink || !overlay || !dialog || !form) {
            console.warn("[THCP] Required elements missing");
            return;
        }

        let resizeTimeout = null;
        let isSubmitting = false;

        function syncBannerHeight() {
            const bannerHeight = banner.offsetHeight;

            if (!bannerHeight) {
                return;
            }

            document.documentElement.style.setProperty("--thcp-banner-height", `${bannerHeight}px`);
            document.documentElement.style.setProperty("--app-banner-height", `${bannerHeight}px`);
            document.documentElement.style.setProperty("--app-top-banner-offset", `${bannerHeight}px`);
            document.body.style.paddingTop = `${bannerHeight}px`;
        }

        function queueBannerSync() {
            window.clearTimeout(resizeTimeout);
            resizeTimeout = window.setTimeout(syncBannerHeight, 10);
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

        function resetPreview() {
            if (imagePreview) {
                imagePreview.src = "";
                hideElement(imagePreview);
            }

            if (imageFeedback) {
                hideElement(imageFeedback);
            }

            if (imageDataInput) {
                imageDataInput.value = "";
            }

            if (fileUpload) {
                fileUpload.value = "";
            }
        }

        function setCurrentPageUrl() {
            if (pageUrlInput) {
                pageUrlInput.value = window.location.href;
            }
        }

        function setSubmittingState(submitting) {
            isSubmitting = submitting;

            if (sendButton) {
                sendButton.disabled = submitting;
                sendButton.textContent = submitting ? "Sending..." : "Send";
            }
        }

        function openPopup(event) {
            if (event) {
                event.preventDefault();
                event.stopPropagation();
            }

            setCurrentPageUrl();
            overlay.hidden = false;
            overlay.setAttribute("aria-hidden", "false");
            document.body.classList.add("comment-popup-open");

            window.setTimeout(function () {
                if (textarea) {
                    textarea.focus();
                } else {
                    dialog.focus();
                }
            }, 0);

            console.log("[THCP] Popup opened");
        }

        function closePopup(event) {
            if (event) {
                event.preventDefault();
                event.stopPropagation();
            }

            if (isSubmitting) {
                return;
            }

            overlay.hidden = true;
            overlay.setAttribute("aria-hidden", "true");
            document.body.classList.remove("comment-popup-open");

            console.log("[THCP] Popup closed");
        }

        syncBannerHeight();
        setCurrentPageUrl();
        overlay.hidden = true;
        overlay.setAttribute("aria-hidden", "true");
        document.body.classList.remove("comment-popup-open");

        window.addEventListener("resize", queueBannerSync);
        window.addEventListener("load", function () {
            syncBannerHeight();
            setCurrentPageUrl();
        });

        if ("ResizeObserver" in window) {
            const resizeObserver = new ResizeObserver(function () {
                queueBannerSync();
            });
            resizeObserver.observe(banner);
        }

        openLink.addEventListener("click", openPopup);

        if (closeX) {
            closeX.addEventListener("click", closePopup);
        }

        if (closeBtn) {
            closeBtn.addEventListener("click", closePopup);
        }

        overlay.addEventListener("click", function (event) {
            if (event.target === overlay) {
                closePopup(event);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && overlay.hidden === false) {
                closePopup(event);
            }
        });

        if (textarea) {
            textarea.addEventListener("paste", function (event) {
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
                            if (imagePreview) {
                                imagePreview.src = loadEvent.target.result;
                                showElement(imagePreview);
                            }

                            if (imageFeedback) {
                                showElement(imageFeedback);
                            }

                            if (imageDataInput) {
                                imageDataInput.value = loadEvent.target.result;
                            }
                        };
                        reader.readAsDataURL(file);
                        break;
                    }
                }
            });
        }

        if (fileUpload) {
            fileUpload.addEventListener("change", function () {
                const file = fileUpload.files && fileUpload.files[0];

                if (!file) {
                    resetPreview();
                    return;
                }

                const reader = new FileReader();
                reader.onload = function (event) {
                    if (imagePreview) {
                        imagePreview.src = event.target.result;
                        showElement(imagePreview);
                    }

                    if (imageFeedback) {
                        showElement(imageFeedback);
                    }

                    if (imageDataInput) {
                        imageDataInput.value = event.target.result;
                    }
                };
                reader.readAsDataURL(file);
            });
        }

        form.addEventListener("submit", async function (event) {
            event.preventDefault();

            if (isSubmitting) {
                return;
            }

            const commentText = textarea ? textarea.value.trim() : "";
            const hasComment = commentText !== "";
            const hasPastedImage = imageDataInput && imageDataInput.value.trim() !== "";
            const hasFile = fileUpload && fileUpload.files && fileUpload.files.length > 0;

            if (!hasComment) {
                window.alert("Please enter a comment before sending.");
                if (textarea) {
                    textarea.focus();
                }
                return;
            }

            setCurrentPageUrl();

            const submitUrl =
                (submitUrlInput && submitUrlInput.value.trim()) ||
                form.getAttribute("action") ||
                "/submit-comment";

            const formData = new FormData(form);

            if (!formData.get("page_url")) {
                formData.set("page_url", window.location.href);
            }

            if (!hasPastedImage && hasFile && fileUpload.files[0]) {
                try {
                    const fileAsDataUrl = await new Promise(function (resolve, reject) {
                        const reader = new FileReader();
                        reader.onload = function (loadEvent) {
                            resolve(loadEvent.target.result);
                        };
                        reader.onerror = function () {
                            reject(new Error("Failed to read uploaded image."));
                        };
                        reader.readAsDataURL(fileUpload.files[0]);
                    });

                    formData.set("imageData", fileAsDataUrl);
                } catch (error) {
                    console.error("[THCP] Failed to convert uploaded file:", error);
                    window.alert("Failed to read the selected image.");
                    return;
                }
            }

            setSubmittingState(true);

            try {
                const response = await fetch(submitUrl, {
                    method: "POST",
                    body: formData,
                    credentials: "same-origin",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest"
                    }
                });

                let result = {};
                try {
                    result = await response.json();
                } catch (jsonError) {
                    throw new Error("Server returned an invalid response.");
                }

                if (!response.ok) {
                    throw new Error(result.error || "Failed to submit comment.");
                }

                console.log("[THCP] Comment submitted successfully");

                form.reset();
                resetPreview();
                setCurrentPageUrl();
                setSubmittingState(false);

                overlay.hidden = true;
                overlay.setAttribute("aria-hidden", "true");
                document.body.classList.remove("comment-popup-open");

                window.alert(result.message || "Comment submitted successfully!");
            } catch (error) {
                console.error("[THCP] Submit failed:", error);
                setSubmittingState(false);
                window.alert(error.message || "Failed to submit comment.");
            }
        });

        console.log("[THCP] Initialized");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initTHCP, { once: true });
    } else {
        initTHCP();
    }
})();