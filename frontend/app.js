/**
 * Agentic AI eBook Grounded RAG Chatbot — Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Determine API base URL dynamically
  const API_BASE =
    window.location.protocol.startsWith("http") && window.location.port === "8000"
      ? window.location.origin
      : "http://127.0.0.1:8000";

  const chatViewport = document.getElementById("chatViewport");
  const welcomeCard = document.getElementById("welcomeCard");
  const messageFeed = document.getElementById("messageFeed");
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const sendBtn = document.getElementById("sendBtn");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  let isSubmitting = false;

  // 1. Health check to update status indicator
  async function checkBackendHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`, { method: "GET" });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "ok") {
          statusDot.className = "status-dot online";
          statusText.textContent = "Backend Online";
          return;
        }
      }
      setOfflineStatus();
    } catch {
      setOfflineStatus();
    }
  }

  function setOfflineStatus() {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Backend Offline";
  }

  checkBackendHealth();
  // Poll health every 30 seconds
  setInterval(checkBackendHealth, 30000);

  // 2. Event Listeners for example prompt chips
  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-query");
      if (query && !isSubmitting) {
        queryInput.value = query;
        handleSubmit();
      }
    });
  });

  // 3. Form submission handling
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSubmit();
  });

  function handleSubmit() {
    const rawQuery = queryInput.value.trim();
    if (!rawQuery || isSubmitting) return;

    // Hide welcome card upon first query
    if (welcomeCard) {
      welcomeCard.style.display = "none";
    }

    // Render User message
    appendUserMessage(rawQuery);

    // Clear input
    queryInput.value = "";
    setInputState(true);

    // Render loading indicator
    const loadingElem = appendLoadingIndicator();

    // Call API
    fetchChatResponse(rawQuery, loadingElem);
  }

  function setInputState(disabled) {
    isSubmitting = disabled;
    queryInput.disabled = disabled;
    sendBtn.disabled = disabled;
    if (!disabled) {
      queryInput.focus();
    }
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  // 4. Render User Message Bubble
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";

    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messageFeed.appendChild(row);
    scrollToBottom();
  }

  // 5. Render Loading Indicator
  function appendLoadingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const loadingCard = document.createElement("div");
    loadingCard.className = "loading-row";
    loadingCard.innerHTML = `
      <span>Synthesizing grounded answer</span>
      <div class="typing-dots">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;

    row.appendChild(loadingCard);
    messageFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  // 6. Call POST /chat
  async function fetchChatResponse(query, loadingElem) {
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        let errorDetail = "An unexpected error occurred.";
        try {
          const errData = await response.json();
          if (errData.detail) errorDetail = errData.detail;
        } catch {
          // ignore parsing error
        }
        loadingElem.remove();
        appendErrorMessage(
          response.status === 500
            ? "An internal server error occurred while processing the request."
            : errorDetail
        );
        setInputState(false);
        return;
      }

      const data = await response.json();
      loadingElem.remove();
      appendAssistantMessage(data);
    } catch {
      loadingElem.remove();
      appendErrorMessage(
        "Unable to connect to the RAG API. Please make sure the backend is running at " +
          API_BASE +
          "."
      );
    } finally {
      setInputState(false);
    }
  }

  // 7. Render Assistant Message with RAG Transparency
  function appendAssistantMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const container = document.createElement("div");
    container.className = "assistant-bubble-container";

    // Text Bubble
    const bubble = document.createElement("div");
    bubble.className = "assistant-bubble";

    // Format text and citations safely
    bubble.innerHTML = formatAssistantText(data.answer);
    container.appendChild(bubble);

    // RAG Transparency Card
    const transparencyCard = document.createElement("div");
    transparencyCard.className = "transparency-card";

    const isGrounded = Boolean(data.is_grounded);
    const confidenceVal = Number(data.confidence || 0).toFixed(4);
    const contextList = Array.isArray(data.context) ? data.context : [];

    const header = document.createElement("div");
    header.className = "transparency-header";

    header.innerHTML = `
      <div class="transparency-badges">
        <span class="grounded-badge ${isGrounded ? "grounded" : "ungrounded"}">
          ${isGrounded ? "Grounded ✓" : "Not enough information in the eBook"}
        </span>
        <span class="confidence-metric">Confidence: ${confidenceVal}</span>
      </div>
      ${
        contextList.length > 0
          ? `<button type="button" class="sources-toggle-btn" aria-expanded="false">
               Sources (${contextList.length} chunks) ▾
             </button>`
          : ""
      }
    `;

    transparencyCard.appendChild(header);

    // If chunks exist, add expandable sources accordion
    if (contextList.length > 0) {
      const sourcesList = document.createElement("div");
      sourcesList.className = "sources-list";
      sourcesList.style.display = "none"; // collapsed by default

      contextList.forEach((chunk, index) => {
        const item = document.createElement("div");
        item.className = "source-item";

        const pageNum = chunk.page !== undefined ? chunk.page : "N/A";
        const scoreVal = chunk.score !== undefined ? Number(chunk.score).toFixed(4) : "0.0000";
        const sourceName = escapeHtml(chunk.source || "Ebook-Agentic-AI.pdf");
        const chunkId = escapeHtml(chunk.chunk_id || `chunk-${index + 1}`);
        const textContent = escapeHtml(chunk.text || "").trim();

        item.innerHTML = `
          <div class="source-item-header">
            <div class="source-item-meta">
              <span class="source-badge">Page ${pageNum}</span>
              <span class="source-badge">${chunkId}</span>
              <span class="source-score">Score: ${scoreVal}</span>
            </div>
            <span>${sourceName}</span>
          </div>
          <div class="source-text">"${textContent}"</div>
        `;
        sourcesList.appendChild(item);
      });

      transparencyCard.appendChild(sourcesList);

      // Toggle behavior
      const toggleBtn = header.querySelector(".sources-toggle-btn");
      if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
          const isExpanded = sourcesList.style.display === "flex";
          sourcesList.style.display = isExpanded ? "none" : "flex";
          toggleBtn.textContent = isExpanded
            ? `Sources (${contextList.length} chunks) ▾`
            : `Sources (${contextList.length} chunks) ▴`;
          toggleBtn.setAttribute("aria-expanded", String(!isExpanded));
          scrollToBottom();
        });
      }
    }

    container.appendChild(transparencyCard);
    row.appendChild(container);
    messageFeed.appendChild(row);
    scrollToBottom();
  }

  // 8. Render Friendly Error Message
  function appendErrorMessage(msg) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const errorBubble = document.createElement("div");
    errorBubble.className = "error-bubble";
    errorBubble.innerHTML = `<strong>Error:</strong> ${escapeHtml(msg)}`;

    row.appendChild(errorBubble);
    messageFeed.appendChild(row);
    scrollToBottom();
  }

  // 9. Format Text and Citations Helper
  function formatAssistantText(text) {
    if (!text) return "";
    let safe = escapeHtml(text);

    // Replace [Page X] with styled citation tags
    safe = safe.replace(/\[Page\s*(\d+)\]/gi, '<span class="citation-tag">[Page $1]</span>');

    // Split paragraphs
    const paragraphs = safe.split(/\n\n+/);
    return paragraphs
      .map((p) => {
        const lines = p.split(/\n/);
        // Detect bullet lists
        if (lines.every((l) => l.trim().startsWith("- ") || l.trim().startsWith("* "))) {
          const items = lines
            .map((l) => `<li>${l.replace(/^[-*]\s+/, "")}</li>`)
            .join("");
          return `<ul>${items}</ul>`;
        }
        return `<p>${lines.join("<br>")}</p>`;
      })
      .join("");
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
