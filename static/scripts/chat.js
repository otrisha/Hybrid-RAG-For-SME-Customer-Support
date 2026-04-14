(() => {
    const messagesEl = document.getElementById("messages");
    const inputEl    = document.getElementById("user-input");
    const sendBtn    = document.getElementById("send-btn");
    const modeSelect = document.getElementById("mode-select");

    // ── Helpers ───────────────────────────────────────────────────────────────

    function timestamp() {
        return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function scrollBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // Convert **bold** and newlines to HTML
    function formatAnswer(text) {
        return escapeHtml(text)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\n/g, "<br>");
    }

    // ── Message builders ──────────────────────────────────────────────────────

    function appendUserMessage(text) {
        const div = document.createElement("div");
        div.className = "message user-message";
        div.innerHTML = `
            <div class="message-label">You</div>
            <div class="message-content">${escapeHtml(text)}</div>
            <div class="message-timestamp">${timestamp()}</div>
        `;
        messagesEl.appendChild(div);
        scrollBottom();
    }

    function appendTyping() {
        const div = document.createElement("div");
        div.className = "message bot-message typing";
        div.innerHTML = `
            <div class="message-label">Support Assistant</div>
            <div class="message-content">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
        `;
        messagesEl.appendChild(div);
        scrollBottom();
        return div;
    }

    function buildSourceTags(sources) {
        if (!sources || sources.length === 0) return "";
        const tags = sources
            .map(s => `<span class="source-tag">${escapeHtml(s)}</span>`)
            .join("");
        return `<div class="sources-row"><span class="sources-label">Sources:</span>${tags}</div>`;
    }

    function buildMeta(data) {
        const parts = [];
        if (data.detected_model) parts.push(`Model: ${escapeHtml(data.detected_model)}`);
        if (data.topic_category) parts.push(`Topic: ${escapeHtml(data.topic_category.replace(/_/g, " "))}`);
        if (data.retrieval_mode) parts.push(`Mode: ${escapeHtml(data.retrieval_mode)}`);
        if (data.latency_seconds != null) parts.push(`${data.latency_seconds}s`);
        return parts.length
            ? `<div class="message-meta">${parts.join(" &nbsp;·&nbsp; ")}</div>`
            : "";
    }

    function appendBotMessage(data) {
        const fallbackNote = data.is_fallback
            ? `<div class="fallback-note">This query could not be fully answered from available documentation.</div>`
            : "";

        const div = document.createElement("div");
        div.className = "message bot-message";
        div.innerHTML = `
            <div class="message-label">Support Assistant</div>
            <div class="message-content">${formatAnswer(data.answer)}</div>
            ${buildSourceTags(data.sources)}
            ${buildMeta(data)}
            ${fallbackNote}
            <div class="message-timestamp">${timestamp()}</div>
        `;
        messagesEl.appendChild(div);
        scrollBottom();
    }

    function appendErrorMessage(text) {
        const div = document.createElement("div");
        div.className = "message bot-message error";
        div.innerHTML = `
            <div class="message-label">Support Assistant</div>
            <div class="message-content">${escapeHtml(text)}</div>
            <div class="message-timestamp">${timestamp()}</div>
        `;
        messagesEl.appendChild(div);
        scrollBottom();
    }

    // ── Send logic ────────────────────────────────────────────────────────────

    function sendMessage() {
        const query = inputEl.value.trim();
        if (!query) return;

        const mode = modeSelect ? modeSelect.value : DEFAULT_MODE;

        appendUserMessage(query);
        inputEl.value = "";
        sendBtn.disabled = true;

        const typing = appendTyping();

        fetch("/chat", {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify({ query, mode }),
        })
        .then(res => res.json())
        .then(data => {
            messagesEl.removeChild(typing);
            if (data.error) {
                const detail = data.detail ? ` — ${data.detail}` : "";
                appendErrorMessage("An error occurred: " + data.error + detail);
            } else {
                appendBotMessage(data);
            }
        })
        .catch(() => {
            messagesEl.removeChild(typing);
            appendErrorMessage("Could not reach the server. Please try again.");
        })
        .finally(() => {
            sendBtn.disabled = false;
            inputEl.focus();
        });
    }

    // ── Event listeners ───────────────────────────────────────────────────────

    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Reload page with selected mode reflected in the URL
    if (modeSelect) {
        modeSelect.addEventListener("change", () => {
            const url = new URL(window.location.href);
            url.searchParams.set("mode", modeSelect.value);
            window.history.replaceState({}, "", url);
        });
    }

    inputEl.focus();
})();
