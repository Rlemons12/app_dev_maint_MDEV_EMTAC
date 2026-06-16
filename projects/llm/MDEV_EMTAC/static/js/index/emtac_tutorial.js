(function () {
    "use strict";

    const STORAGE_KEY = "emtac_index_tutorial_seen_v10_conversation_toggle_step";
    const LANGUAGE_STORAGE_KEY = "emtac_index_tutorial_language";
    const HIGHLIGHT_CLASS = "emtac-tutorial-highlight";

    let currentLanguage = localStorage.getItem(LANGUAGE_STORAGE_KEY) || "en";

    const translations = {
        en: {
            back: "Back",
            next: "Next",
            finish: "Finish",
            step: "Step",
            of: "of",
            languageLabel: "Language",
            english: "English",
            japanese: "Japanese"
        },
        ja: {
            back: "戻る",
            next: "次へ",
            finish: "終了",
            step: "ステップ",
            of: "/",
            languageLabel: "言語",
            english: "英語",
            japanese: "日本語"
        }
    };

    const tutorialSteps = [
        {
            selector: ".five331-partial-template",
            demo: true,
            en: {
                title: "Welcome to EMTAC Assistant",
                text: "This tour shows the sidebar, quick actions, conversation memory toggle, question box, answer area, supporting panels, and pop-out viewers."
            },
            ja: {
                title: "EMTACアシスタントへようこそ",
                text: "このチュートリアルでは、サイドバー、クイック操作、会話メモリ切替、質問欄、回答欄、関連パネル、ポップアウトビューアの使い方を説明します。"
            }
        },
        {
            selector: "#mainSidebar",
            en: {
                title: "Main Sidebar",
                text: "This is your main navigation area. Use it to move between EMTAC pages and relaunch this tutorial."
            },
            ja: {
                title: "メインサイドバー",
                text: "ここはメインのナビゲーションエリアです。EMTAC内の各ページへ移動したり、このチュートリアルを再表示できます。"
            }
        },
        {
            selector: ".five331-sidebar-toggle",
            openChatSidebar: true,
            en: {
                title: "Quick Actions Slide-Out",
                text: "This button opens the assistant slide-out panel with Clear Chat, Conversation ON/OFF, Help, feedback, rating, and voice controls."
            },
            ja: {
                title: "クイック操作スライドパネル",
                text: "このボタンを押すと、チャットのクリア、会話メモリのオン／オフ、ヘルプ、フィードバック、評価、音声操作があるスライドパネルが開きます。"
            }
        },
        {
            selector: ".five331-quick-actions",
            openChatSidebar: true,
            en: {
                title: "Quick Actions",
                text: "Use Quick Actions to clear the chat, turn conversation memory on or off, and open Help."
            },
            ja: {
                title: "クイック操作",
                text: "クイック操作では、チャットのクリア、会話メモリのオン／オフ、ヘルプの表示ができます。"
            }
        },
        {
            selector: "#conversation-mode-toggle",
            openChatSidebar: true,
            en: {
                title: "Conversation Memory Toggle",
                text: "Conversation ON lets the assistant remember the current chat for follow-up questions. Conversation OFF uses single-turn mode, so every question is treated as a new topic."
            },
            ja: {
                title: "会話メモリ切替",
                text: "Conversation ONでは、アシスタントが現在のチャット内容を使ってフォローアップ質問に対応できます。Conversation OFFでは単発モードになり、各質問が新しい話題として扱われます。"
            }
        },
        {
            selector: ".five331-rating-section",
            openChatSidebar: true,
            en: {
                title: "Feedback and Rating",
                text: "Use this section to leave comments, corrections, suggestions, or a rating for the answer."
            },
            ja: {
                title: "フィードバックと評価",
                text: "このエリアでは、回答へのコメント、修正、提案、評価を入力できます。"
            }
        },
        {
            selector: ".five331-chat-input",
            en: {
                title: "Question Box",
                text: "Type the maintenance or troubleshooting question here. Specific questions usually produce better answers."
            },
            ja: {
                title: "質問欄",
                text: "ここに保全やトラブルシューティングに関する質問を入力します。具体的な質問ほど、より良い回答につながります。"
            }
        },
        {
            selector: "#chat-area",
            en: {
                title: "Answer Area",
                text: "The answer appears here first. Supporting payload items may load afterward on the right."
            },
            ja: {
                title: "回答エリア",
                text: "回答は最初にここに表示されます。その後、右側に関連する資料、画像、図面、部品情報が表示される場合があります。"
            }
        },
        {
            selector: "#images-container",
            openPanelSelector: "#images-container",
            en: {
                title: "Images Panel",
                text: "Images show visual references connected to the answer."
            },
            ja: {
                title: "画像パネル",
                text: "画像パネルには、回答に関連する視覚的な参考情報が表示されます。"
            }
        },
        {
            selector: "#emtac-global-image-viewer-overlay .emtac-viewer-header, #thumbnails-section .image-document-link, #thumbnails-section .document-chunk-link",
            openPanelSelector: "#images-container",
            openDemoImageViewer: true,
            skipHighlightScroll: true,
            en: {
                title: "Image Pop-Out Viewer",
                text: "Clicking an image result opens a full in-page image viewer. Use this viewer to inspect visual references without leaving the assistant page."
            },
            ja: {
                title: "画像ポップアウトビューア",
                text: "画像結果をクリックすると、ページ内の画像ビューアが開きます。アシスタント画面を離れずに画像を確認できます。"
            }
        },
        {
            selector: "#documents-container",
            openPanelSelector: "#documents-container",
            en: {
                title: "Documents Panel",
                text: "Documents show manuals, procedures, training files, or source chunks related to the answer."
            },
            ja: {
                title: "ドキュメントパネル",
                text: "ドキュメントパネルには、回答に関連するマニュアル、手順書、トレーニング資料、文書の一部が表示されます。"
            }
        },
        {
            selector: "#emtac-global-text-viewer-overlay .emtac-viewer-header, #emtac-doc-popout-overlay .emtac-doc-popout-header, #doc-links-section .document-popout-link, #doc-links-section .document-chunk-link",
            openPanelSelector: "#documents-container",
            openDemoDocumentViewer: true,
            skipHighlightScroll: true,
            en: {
                title: "Document Pop-Out Viewer",
                text: "Document results can open in a full in-page viewer so users can read source material more easily."
            },
            ja: {
                title: "ドキュメントポップアウトビューア",
                text: "ドキュメント結果はページ内ビューアで開くことができ、元資料を読みやすく確認できます。"
            }
        },
        {
            selector: "#drawings-container",
            openPanelSelector: "#drawings-container",
            en: {
                title: "Drawings Panel",
                text: "Drawings show related prints, assemblies, or drawing navigation when available."
            },
            ja: {
                title: "図面パネル",
                text: "図面パネルには、関連する図面、組立図、利用可能な図面ナビゲーションが表示されます。"
            }
        },
        {
            selector: "#emtac-drawing-viewer-overlay .emtac-drawing-viewer-header, #drawing-section .drawing-link-button, #drawing-section .document-chunk-link",
            openPanelSelector: "#drawings-container",
            openDemoDrawingViewer: true,
            skipHighlightScroll: true,
            en: {
                title: "Drawing Pop-Out Viewer",
                text: "Drawing results open in a drawing detail viewer with drawing metadata and a PDF preview when a drawing file is available."
            },
            ja: {
                title: "図面ポップアウトビューア",
                text: "図面結果を開くと、図面の詳細情報や、図面ファイルがある場合はPDFプレビューを確認できます。"
            }
        },
        {
            selector: "#parts-container",
            openPanelSelector: "#parts-container",
            en: {
                title: "Parts Panel",
                text: "Parts show related components or BOM information."
            },
            ja: {
                title: "部品パネル",
                text: "部品パネルには、関連部品やBOM情報が表示されます。"
            }
        },
        {
            selector: "#emtac-part-viewer-overlay .emtac-part-viewer-header, #parts-container .part-link-button, #parts-container .document-chunk-link",
            openPanelSelector: "#parts-container",
            openDemoPartViewer: true,
            skipHighlightScroll: true,
            en: {
                title: "Part Pop-Out Viewer",
                text: "Part results open in a part detail viewer showing part information, related images, and related drawings."
            },
            ja: {
                title: "部品ポップアウトビューア",
                text: "部品結果を開くと、部品情報、関連画像、関連図面を確認できる詳細ビューアが表示されます。"
            }
        },
        {
            selector: "#emtacTutorialSidebarLink",
            en: {
                title: "Finished",
                text: "You can reopen this tutorial any time from the Tutorial link in the sidebar."
            },
            ja: {
                title: "完了",
                text: "このチュートリアルは、サイドバーのTutorialリンクからいつでも再表示できます。"
            }
        }
    ];

    let currentStep = 0;
    let previousHighlightedElement = null;
    let eventsBound = false;
    let stepRestoreActions = [];

    function byId(id) {
        return document.getElementById(id);
    }

    function t(key) {
        return translations[currentLanguage][key] || translations.en[key] || key;
    }

    function getStepText(step) {
        return step[currentLanguage] || step.en;
    }

    function getElements() {
        return {
            overlay: byId("emtacTutorialOverlay"),
            card: document.querySelector(".emtac-tutorial-card"),
            title: byId("emtacTutorialTitle"),
            text: byId("emtacTutorialText"),
            closeBtn: byId("emtacTutorialCloseBtn"),
            backBtn: byId("emtacTutorialBackBtn"),
            nextBtn: byId("emtacTutorialNextBtn"),
            stepText: byId("emtacTutorialStepText"),
            sidebarLink: byId("emtacTutorialSidebarLink")
        };
    }

    function ensureLanguageSelector() {
        const el = getElements();

        if (!el.card || byId("emtacTutorialLanguageSelect")) {
            return;
        }

        const header = el.card.querySelector(".emtac-tutorial-header");

        if (!header) {
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.className = "emtac-tutorial-language-wrapper";

        const label = document.createElement("label");
        label.setAttribute("for", "emtacTutorialLanguageSelect");
        label.textContent = t("languageLabel") + ":";

        const select = document.createElement("select");
        select.id = "emtacTutorialLanguageSelect";
        select.className = "emtac-tutorial-language-select";

        const english = document.createElement("option");
        english.value = "en";
        english.textContent = "English";

        const japanese = document.createElement("option");
        japanese.value = "ja";
        japanese.textContent = "日本語";

        select.appendChild(english);
        select.appendChild(japanese);
        select.value = currentLanguage;

        select.addEventListener("change", function () {
            currentLanguage = select.value || "en";
            localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
            renderStep();
        });

        wrapper.appendChild(label);
        wrapper.appendChild(select);

        header.insertBefore(wrapper, el.closeBtn);
    }

    function updateLanguageSelectorText() {
        const wrapper = document.querySelector(".emtac-tutorial-language-wrapper");
        const label = wrapper ? wrapper.querySelector("label") : null;
        const select = byId("emtacTutorialLanguageSelect");

        if (label) {
            label.textContent = t("languageLabel") + ":";
        }

        if (select) {
            select.value = currentLanguage;
        }
    }

    function moveTutorialOverlayToBody() {
        const overlay = byId("emtacTutorialOverlay");

        if (!overlay || !document.body) {
            return;
        }

        if (overlay.parentElement !== document.body) {
            document.body.appendChild(overlay);
        }
    }

    function isTutorialOpen() {
        const el = getElements();
        return Boolean(el.overlay && el.overlay.style.display !== "none");
    }

    function rememberRestoreAction(action) {
        if (typeof action === "function") {
            stepRestoreActions.push(action);
        }
    }

    function restoreStepState() {
        while (stepRestoreActions.length > 0) {
            const action = stepRestoreActions.pop();

            try {
                action();
            } catch (error) {
                console.warn("[EMTAC Tutorial] Failed to restore step state.", error);
            }
        }

        moveTutorialOverlayToBody();
    }

    function getStepTarget(step) {
        if (!step || !step.selector) {
            return null;
        }

        return document.querySelector(step.selector);
    }

    function loadTutorialDemoContent() {
        const input = byId("user_input");
        const answer = byId("answer");
        const docs = byId("doc-links-section");
        const images = byId("thumbnails-section");
        const drawings = byId("drawing-section");
        const parts = byId("parts-container");

        if (input && !input.value.trim()) {
            input.value = currentLanguage === "ja"
                ? "例：バッグロードモニター速度の問題の原因は何ですか？"
                : "Example: What causes a bag load monitor speed issue?";
        }

        if (answer) {
            answer.innerHTML = currentLanguage === "ja"
                ? `
                    <strong>回答例:</strong><br>
                    バッグロードモニター速度の問題は、レシピ速度設定が高すぎる、バッグの位置ずれ、
                    機械的な抵抗、グリッパーのタイミング、ロボット動作設定などが原因になる場合があります。
                `
                : `
                    <strong>Example Answer:</strong><br>
                    Bag load monitor speed issues may be caused by a recipe speed setting that is too high,
                    bag alignment problems, mechanical drag, gripper timing, or robot motion settings.
                `;
        }

        if (docs) {
            docs.innerHTML = currentLanguage === "ja"
                ? `
                    <div class="document-item emtac-tutorial-demo-item">
                        <div class="document-header-row">
                            <button type="button" class="document-chunk-link">
                                例：バッグロードモニター速度手順書
                            </button>
                            <button type="button" class="document-popout-link">開く</button>
                        </div>
                        <div class="document-chunk-text" style="display:block;">
                            ドキュメントは、回答を元資料で確認するために使用します。
                        </div>
                    </div>
                `
                : `
                    <div class="document-item emtac-tutorial-demo-item">
                        <div class="document-header-row">
                            <button type="button" class="document-chunk-link">
                                Example Document: Bag Load Monitor Speed Procedure
                            </button>
                            <button type="button" class="document-popout-link">Open</button>
                        </div>
                        <div class="document-chunk-text" style="display:block;">
                            Documents help verify the answer against source material.
                        </div>
                    </div>
                `;
        }

        if (images) {
            images.innerHTML = currentLanguage === "ja"
                ? `
                    <div class="document-item image-document-item emtac-tutorial-demo-item">
                        <button type="button" class="document-chunk-link image-document-link">
                            例：ロードステーション バッグ位置合わせ画像
                        </button>
                        <div class="document-chunk-text" style="display:block;">
                            画像は、機械エリアやセットアップ状態を視覚的に確認するために使用します。
                        </div>
                    </div>
                `
                : `
                    <div class="document-item image-document-item emtac-tutorial-demo-item">
                        <button type="button" class="document-chunk-link image-document-link">
                            Example Image: Load Station Bag Alignment
                        </button>
                        <div class="document-chunk-text" style="display:block;">
                            Images help visually confirm machine areas and setup conditions.
                        </div>
                    </div>
                `;
        }

        if (drawings) {
            drawings.innerHTML = currentLanguage === "ja"
                ? `
                    <div class="drawing-item emtac-tutorial-demo-item">
                        <button type="button" class="drawing-link-button document-chunk-link">
                            <strong>例：ロードエンドインバーター組立図</strong>
                        </button>
                        <div class="drawing-meta">
                            <div><b>番号:</b> DWG-TUTORIAL-001</div>
                            <div><b>エリア:</b> サンプルエリア</div>
                            <div><b>モデル:</b> サンプルモデル</div>
                            <div><b>資産番号:</b> サンプル資産</div>
                        </div>
                    </div>
                `
                : `
                    <div class="drawing-item emtac-tutorial-demo-item">
                        <button type="button" class="drawing-link-button document-chunk-link">
                            <strong>Example Drawing: Load End Inverter Assembly</strong>
                        </button>
                        <div class="drawing-meta">
                            <div><b>No:</b> DWG-TUTORIAL-001</div>
                            <div><b>Area:</b> Example Area</div>
                            <div><b>Model:</b> Example Model</div>
                            <div><b>Asset Number:</b> Example Asset</div>
                        </div>
                    </div>
                `;
        }

        if (parts) {
            parts.innerHTML = currentLanguage === "ja"
                ? `
                    <div class="document-item part-item emtac-tutorial-demo-item">
                        <button type="button" class="document-chunk-link part-link-button">
                            バッグロードグリッパー組立 — TUTORIAL-001
                        </button>
                        <div class="document-chunk-text" style="display:block;">
                            部品とBOM結果は、交換部品の識別に役立ちます。
                        </div>
                    </div>
                `
                : `
                    <div class="document-item part-item emtac-tutorial-demo-item">
                        <button type="button" class="document-chunk-link part-link-button">
                            Bag Load Gripper Assembly — TUTORIAL-001
                        </button>
                        <div class="document-chunk-text" style="display:block;">
                            Parts and BOM results help identify replacement components.
                        </div>
                    </div>
                `;
        }
    }

    function toggleFive331Sidebar() {
        const sidebar = byId("five331-sidebar");

        if (!sidebar) {
            return;
        }

        if (typeof window.five331ToggleSidebar === "function") {
            window.five331ToggleSidebar();
            return;
        }

        sidebar.classList.toggle("five331-closed");
    }

    function openChatSidebarIfNeeded(step) {
        if (!step || !step.openChatSidebar) {
            return;
        }

        const sidebar = byId("five331-sidebar");

        if (!sidebar) {
            return;
        }

        const wasClosed = sidebar.classList.contains("five331-closed");

        if (wasClosed) {
            toggleFive331Sidebar();

            rememberRestoreAction(function () {
                const isOpen = !sidebar.classList.contains("five331-closed");

                if (isOpen) {
                    toggleFive331Sidebar();
                }
            });
        }
    }

    function openPanelIfNeeded(step) {
        if (!step || !step.openPanelSelector) {
            return;
        }

        const panelBody = document.querySelector(step.openPanelSelector);

        if (!panelBody) {
            return;
        }

        const panel = panelBody.closest(".five331-panel");

        if (!panel) {
            return;
        }

        const wasCollapsed = panel.classList.contains("is-collapsed");

        if (wasCollapsed) {
            panel.classList.remove("is-collapsed");

            const toggleBtn = panel.querySelector(".toggle-btn");

            if (toggleBtn) {
                toggleBtn.setAttribute("aria-expanded", "true");
                toggleBtn.textContent = currentLanguage === "ja" ? "切替" : "Toggle";
            }

            rememberRestoreAction(function () {
                panel.classList.add("is-collapsed");

                if (toggleBtn) {
                    toggleBtn.setAttribute("aria-expanded", "false");
                    toggleBtn.textContent = currentLanguage === "ja" ? "開く" : "Open";
                }
            });
        }
    }

    function openDemoViewerIfNeeded(step) {
        if (!step) {
            return;
        }

        if (step.openDemoImageViewer) {
            openTutorialImageViewer();
            return;
        }

        if (step.openDemoDocumentViewer) {
            openTutorialDocumentViewer();
            return;
        }

        if (step.openDemoDrawingViewer) {
            openTutorialDrawingViewer();
            return;
        }

        if (step.openDemoPartViewer) {
            openTutorialPartViewer();
        }
    }

    function openTutorialImageViewer() {
        const title = currentLanguage === "ja" ? "チュートリアル画像例" : "Tutorial Example Image";

        if (typeof window.openImageViewerInPage === "function") {
            window.openImageViewerInPage(title, "/static/background_images/abstractfractal.jpg");
        } else if (typeof window.openImageViewer === "function") {
            window.openImageViewer(title, "/static/background_images/abstractfractal.jpg");
        } else {
            console.warn("[EMTAC Tutorial] No image viewer function found.");
            return;
        }

        moveTutorialOverlayToBody();

        rememberRestoreAction(function () {
            removeById("emtac-global-image-viewer-overlay");
            document.body.classList.remove("emtac-viewer-open");
            moveTutorialOverlayToBody();
        });
    }

    function openTutorialDocumentViewer() {
        const title = currentLanguage === "ja" ? "チュートリアル文書例" : "Tutorial Example Document";
        const text = currentLanguage === "ja"
            ? "ここでは、全文書または選択された文書チャンクを確認できます。アシスタント画面を離れずに元資料を読むことができます。"
            : "This is where the full document or selected document chunk opens for review. Use the document viewer to read source material without leaving the assistant page.";

        if (typeof window.openDocumentViewer === "function") {
            window.openDocumentViewer(title, text);
        } else if (typeof window.openTextViewer === "function") {
            window.openTextViewer(title, text);
        } else if (typeof window.openChunkPopout === "function") {
            window.openChunkPopout(title, text);
        } else {
            console.warn("[EMTAC Tutorial] No document viewer function found.");
            return;
        }

        moveTutorialOverlayToBody();

        rememberRestoreAction(function () {
            removeById("emtac-global-text-viewer-overlay");
            removeById("emtac-doc-popout-overlay");
            document.body.classList.remove("emtac-viewer-open");
            document.body.classList.remove("emtac-doc-popout-open");
            moveTutorialOverlayToBody();
        });
    }

    function openTutorialDrawingViewer() {
        if (typeof window.openDrawingDetailsInPage !== "function") {
            console.warn("[EMTAC Tutorial] No drawing viewer function found.");
            return;
        }

        window.openDrawingDetailsInPage({
            id: "tutorial-drawing",
            drw_name: currentLanguage === "ja" ? "チュートリアル図面例" : "Tutorial Example Drawing",
            drw_number: "DWG-TUTORIAL-001",
            drw_revision: "A",
            drw_equipment_name: currentLanguage === "ja" ? "サンプル設備" : "Example Equipment",
            _area: currentLanguage === "ja" ? "サンプルエリア" : "Example Area",
            _model: currentLanguage === "ja" ? "サンプルモデル" : "Example Model",
            _asset: currentLanguage === "ja" ? "サンプル資産" : "Example Asset",
            spare_parts: [
                {
                    part_number: "TUTORIAL-001",
                    name: currentLanguage === "ja" ? "バッグロードグリッパー組立" : "Bag Load Gripper Assembly"
                }
            ]
        });

        moveTutorialOverlayToBody();

        rememberRestoreAction(function () {
            removeById("emtac-drawing-viewer-overlay");
            document.body.classList.remove("emtac-drawing-viewer-open");
            moveTutorialOverlayToBody();
        });
    }

    function openTutorialPartViewer() {
        if (typeof window.openPartDetailsInPage !== "function") {
            console.warn("[EMTAC Tutorial] No part viewer function found.");
            return;
        }

        const partName = currentLanguage === "ja" ? "バッグロードグリッパー組立" : "Bag Load Gripper Assembly";

        window.openPartDetailsInPage({
            part: {
                id: "tutorial-part",
                name: partName,
                part_name: partName,
                part_number: "TUTORIAL-001",
                manufacturer: currentLanguage === "ja" ? "サンプルメーカー" : "Example Manufacturer",
                model: currentLanguage === "ja" ? "サンプルモデル" : "Example Model",
                description: currentLanguage === "ja"
                    ? "チュートリアル中に表示される部品詳細の例です。"
                    : "Example part detail shown during the tutorial."
            },
            label: `${partName} — TUTORIAL-001`,
            images: [],
            drawings: [
                {
                    title: currentLanguage === "ja" ? "チュートリアル図面例" : "Tutorial Example Drawing",
                    number: "DWG-TUTORIAL-001",
                    revision: "A",
                    url: ""
                }
            ],
            errorMessage: ""
        });

        moveTutorialOverlayToBody();

        rememberRestoreAction(function () {
            removeById("emtac-part-image-zoom-overlay");
            removeById("emtac-part-viewer-overlay");
            document.body.classList.remove("emtac-part-viewer-open");
            moveTutorialOverlayToBody();
        });
    }

    function removeById(id) {
        const el = byId(id);

        if (el) {
            el.remove();
        }
    }

    function clearHighlight() {
        if (previousHighlightedElement) {
            previousHighlightedElement.classList.remove(HIGHLIGHT_CLASS);
            previousHighlightedElement = null;
        }
    }

    function highlightTarget(target, skipScroll) {
        clearHighlight();

        if (!target) {
            return;
        }

        target.classList.add(HIGHLIGHT_CLASS);
        previousHighlightedElement = target;

        if (skipScroll) {
            return;
        }

        setTimeout(function () {
            try {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest",
                    inline: "nearest"
                });
            } catch (error) {
                target.scrollIntoView(false);
            }

            window.scrollTo({
                left: 0,
                top: window.scrollY,
                behavior: "instant"
            });
        }, 50);
    }

    function moveCardNearTarget() {
        const el = getElements();

        if (!el.card) {
            return;
        }

        const viewerIsOpen =
            byId("emtac-global-image-viewer-overlay") ||
            byId("emtac-global-text-viewer-overlay") ||
            byId("emtac-doc-popout-overlay") ||
            byId("emtac-drawing-viewer-overlay") ||
            byId("emtac-part-viewer-overlay");

        el.card.classList.remove(
            "emtac-tutorial-card-left",
            "emtac-tutorial-card-right",
            "emtac-tutorial-card-center"
        );

        el.card.style.left = "50%";
        el.card.style.right = "auto";
        el.card.style.top = "auto";
        el.card.style.bottom = viewerIsOpen ? "40px" : "150px";
        el.card.style.transform = "translateX(-50%)";
    }

    function renderStep() {
        const el = getElements();
        const step = tutorialSteps[currentStep];

        if (!el.overlay || !step) {
            return;
        }

        if (step.demo) {
            loadTutorialDemoContent();
        }

        moveTutorialOverlayToBody();

        openChatSidebarIfNeeded(step);
        openPanelIfNeeded(step);
        openDemoViewerIfNeeded(step);

        moveTutorialOverlayToBody();
        ensureLanguageSelector();

        const activeText = getStepText(step);
        const target = getStepTarget(step);

        if (el.title) {
            el.title.textContent = activeText.title;
        }

        if (el.text) {
            el.text.textContent = activeText.text;
        }

        if (el.stepText) {
            el.stepText.textContent = `${t("step")} ${currentStep + 1} ${t("of")} ${tutorialSteps.length}`;
        }

        if (el.backBtn) {
            el.backBtn.textContent = t("back");
            el.backBtn.disabled = currentStep === 0;
        }

        if (el.nextBtn) {
            el.nextBtn.textContent = currentStep === tutorialSteps.length - 1 ? t("finish") : t("next");
        }

        updateLanguageSelectorText();
        highlightTarget(target, Boolean(step.skipHighlightScroll));
        moveCardNearTarget();
    }

    function openTutorial(forceStart) {
        moveTutorialOverlayToBody();

        const el = getElements();

        if (!el.overlay) {
            console.warn("[EMTAC Tutorial] Missing #emtacTutorialOverlay.");
            return;
        }

        restoreStepState();
        clearHighlight();

        if (forceStart) {
            currentStep = 0;
        }

        loadTutorialDemoContent();

        el.overlay.style.display = "flex";
        el.overlay.setAttribute("aria-hidden", "false");

        renderStep();
    }

    function closeTutorial(markSeen) {
        const el = getElements();

        restoreStepState();
        clearHighlight();

        if (el.overlay) {
            el.overlay.style.display = "none";
            el.overlay.setAttribute("aria-hidden", "true");
        }

        if (markSeen) {
            localStorage.setItem(STORAGE_KEY, "true");
        }
    }

    function nextStep(event) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }

        restoreStepState();

        if (currentStep >= tutorialSteps.length - 1) {
            closeTutorial(true);
            return;
        }

        currentStep += 1;
        renderStep();
    }

    function previousStep(event) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }

        restoreStepState();

        if (currentStep <= 0) {
            return;
        }

        currentStep -= 1;
        renderStep();
    }

    function bindEvents() {
        if (eventsBound) {
            return;
        }

        eventsBound = true;

        moveTutorialOverlayToBody();

        const el = getElements();

        if (el.nextBtn) {
            el.nextBtn.addEventListener("click", nextStep);
        }

        if (el.backBtn) {
            el.backBtn.addEventListener("click", previousStep);
        }

        if (el.closeBtn) {
            el.closeBtn.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();
                closeTutorial(true);
            });
        }

        if (el.sidebarLink) {
            el.sidebarLink.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();
                openTutorial(true);
            });
        }

        document.addEventListener("keydown", function (event) {
            if (!isTutorialOpen()) {
                return;
            }

            if (event.key === "Escape") {
                closeTutorial(true);
            }

            if (event.key === "ArrowRight") {
                nextStep(event);
            }

            if (event.key === "ArrowLeft") {
                previousStep(event);
            }
        });

        window.addEventListener("resize", function () {
            if (!isTutorialOpen()) {
                return;
            }

            renderStep();
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        bindEvents();
    });

    window.EMTACTutorial = {
        open: function () {
            openTutorial(true);
        },
        reset: function () {
            localStorage.removeItem(STORAGE_KEY);
            localStorage.removeItem("emtac_index_tutorial_seen_v8");
            localStorage.removeItem("emtac_index_tutorial_seen_v9_conversation_toggle");
            localStorage.removeItem(LANGUAGE_STORAGE_KEY);
            currentLanguage = "en";
        },
        close: function () {
            closeTutorial(false);
        },
        setLanguage: function (language) {
            currentLanguage = language === "ja" ? "ja" : "en";
            localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
            renderStep();
        }
    };
})();