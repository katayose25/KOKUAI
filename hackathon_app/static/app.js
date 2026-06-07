(function () {
  let currentTurnText = null;
  let currentTurnRow = null;
  let liveSource = null;
  let syncMode = false;
  const typewriterTimers = new WeakMap();

  function listAtBottom(list) {
    return list.scrollHeight - list.scrollTop - list.clientHeight < 80;
  }

  function scrollListToBottom(list) {
    list.scrollTop = list.scrollHeight;
  }

  function scrollRowIntoList(list, row) {
    const rowTop = row.offsetTop - list.offsetTop;
    const rowBot = rowTop + row.offsetHeight;
    const viewTop = list.scrollTop;
    const viewBot = list.scrollTop + list.clientHeight;
    if (rowTop < viewTop) {
      list.scrollTo({ top: rowTop, behavior: "smooth" });
    } else if (rowBot > viewBot) {
      list.scrollTo({ top: rowBot - list.clientHeight, behavior: "smooth" });
    }
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>'"]/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c];
    });
  }

  function setStatus(text, state) {
    const el = document.getElementById("stream-status");
    if (!el) return;
    el.textContent = text;
    el.classList.remove("running", "done", "error");
    if (state) el.classList.add(state);
  }

  function setModelStatus(text, ready) {
    const el = document.getElementById("model-status");
    if (!el) return;
    el.textContent = text;
    el.classList.remove("running", "done", "error");
    el.classList.add(ready ? "done" : "running");
  }

  async function pollModelStatus() {
    const startButton = document.getElementById("start-live-asr");
    const soapButton = document.getElementById("create-soap");
    if (!startButton && !soapButton) return;
    try {
      const res = await fetch("/api/model-status", { cache: "no-store" });
      const data = await res.json();
      const ready = data.asr === "ready" && data.chart === "ready";
      if (ready) {
        if (startButton) startButton.disabled = false;
        if (soapButton) soapButton.disabled = false;
        setModelStatus("モデル準備完了", true);
      } else {
        if (startButton) startButton.disabled = true;
        if (soapButton) soapButton.disabled = true;
        setModelStatus(`モデル準備中 ASR:${data.asr} Chart:${data.chart}`, false);
        setTimeout(pollModelStatus, 3000);
      }
    } catch (_) {
      setModelStatus("モデル状態を確認中...", false);
      setTimeout(pollModelStatus, 3000);
    }
  }

  function typewriterSpeed() {
    return parseInt(document.body.dataset.typewriterSpeed || "70", 10);
  }

  function transcriptRevealDelaySec() {
    return parseFloat(document.body.dataset.transcriptRevealDelay || "1.2");
  }

  function ensureFullText(row) {
    const p = row ? row.querySelector("p") : null;
    if (!row || !p) return "";
    if (row.dataset.fullText === undefined) row.dataset.fullText = p.textContent || "";
    return row.dataset.fullText || "";
  }

  function stopTypewriter(row) {
    const timer = typewriterTimers.get(row);
    if (timer) {
      clearInterval(timer);
      typewriterTimers.delete(row);
    }
  }

  function startTypewriter(row) {
    const p = row ? row.querySelector("p") : null;
    if (!row || !p || row.dataset.typed === "1") return;
    ensureFullText(row);
    stopTypewriter(row);
    row.dataset.typed = "1";
    p.textContent = "";
    let idx = 0;
    const timer = setInterval(function () {
      const full = row.dataset.fullText || "";
      if (idx < full.length) {
        idx += 1;
        p.textContent = full.slice(0, idx);
      }
      if (idx >= full.length && row.dataset.complete === "1") {
        stopTypewriter(row);
      }
    }, typewriterSpeed());
    typewriterTimers.set(row, timer);
  }

  function hideTypewriterRow(row) {
    const p = row ? row.querySelector("p") : null;
    if (!row || !p) return;
    ensureFullText(row);
    stopTypewriter(row);
    row.dataset.typed = "0";
    p.textContent = "";
  }

  function beginTurn(data) {
    const list = document.getElementById("transcript-list");
    if (!list) return;
    list.querySelectorAll(".empty-state").forEach((node) => node.remove());
    const row = document.createElement("article");
    row.className = `turn-row ${data.speaker || "unknown"}`;
    row.dataset.start = String(data.start || 0);
    row.dataset.end = "0";
    row.dataset.fullText = "";
    row.dataset.typed = "0";
    row.dataset.complete = "0";
    row.innerHTML = `<span>${escapeHtml(data.speaker_label || "不明")}</span><p></p>`;
    list.appendChild(row);
    currentTurnRow = row;
    currentTurnText = row.querySelector("p");
    if (syncMode) {
      syncTranscriptAt(currentAudioTime());
    } else {
      if (listAtBottom(list)) scrollListToBottom(list);
    }
  }

  function appendToken(data) {
    const list = document.getElementById("transcript-list");
    if (!currentTurnText || !list || !currentTurnRow) return;
    const piece = data.text || "";
    currentTurnRow.dataset.fullText = (currentTurnRow.dataset.fullText || "") + piece;
    if (syncMode) {
      syncTranscriptAt(currentAudioTime());
    } else {
      currentTurnText.textContent += piece;
      if (listAtBottom(list)) scrollListToBottom(list);
    }
  }

  function endTurn(data) {
    if (currentTurnRow && data && data.end !== undefined) {
      currentTurnRow.dataset.end = String(data.end || 0);
      currentTurnRow.dataset.complete = "1";
    }
    if (syncMode) syncTranscriptAt(currentAudioTime());
    currentTurnText = null;
    currentTurnRow = null;
  }

  function currentEncounterAudio() {
    return document.querySelector("audio.encounter-audio");
  }

  function currentAudioTime() {
    const audio = currentEncounterAudio();
    return audio ? (audio.currentTime || 0) : 0;
  }

  function syncTranscriptAt(time) {
    let latestVisibleRow = null;
    document.querySelectorAll(".turn-row[data-start]").forEach(function (row) {
      const start = parseFloat(row.dataset.start || "0");
      const visible = Number.isFinite(start) && start + transcriptRevealDelaySec() <= time;
      row.classList.toggle("sync-hidden", !visible);
      row.classList.toggle("sync-revealed", visible);
      if (visible) {
        latestVisibleRow = row;
        startTypewriter(row);
      } else {
        hideTypewriterRow(row);
      }
    });
    const list = document.getElementById("transcript-list");
    const activeRow = document.querySelector(".turn-row.active");
    const scrollTarget = activeRow || latestVisibleRow;
    if (scrollTarget && list) scrollRowIntoList(list, scrollTarget);
  }

  function hasTimestampedTurns() {
    return Array.from(document.querySelectorAll(".turn-row[data-start][data-end]")).some(function (row) {
      const start = parseFloat(row.dataset.start || "0");
      const end = parseFloat(row.dataset.end || "0");
      return Number.isFinite(start) && Number.isFinite(end) && end > start;
    });
  }

  function activateSyncMode(audio) {
    syncMode = true;
    syncTranscriptAt(audio ? (audio.currentTime || 0) : currentAudioTime());
  }

  function preparePersistedTranscriptRows() {
    document.querySelectorAll(".turn-row[data-start]").forEach(function (row) {
      const p = row.querySelector("p");
      if (!p) return;
      if (row.dataset.fullText === undefined) row.dataset.fullText = p.textContent || "";
      if (row.dataset.typed === undefined) row.dataset.typed = "0";
      if (row.dataset.complete === undefined) row.dataset.complete = "1";
    });
  }

  function initializeTranscriptReveal(root) {
    // Only prepare rows for sync — do NOT auto-activate sync mode.
    // Sync mode activates only when the user actually plays the audio.
    preparePersistedTranscriptRows();
  }

  function trySyncWithPlayingAudio() {
    const audioEl = document.querySelector("audio.encounter-audio");
    if (audioEl && !audioEl.paused) activateSyncMode(audioEl);
  }

  function deactivateSyncMode() {
    syncMode = false;
    document.querySelectorAll(".turn-row").forEach(function (row) {
      const p = row.querySelector("p");
      stopTypewriter(row);
      if (p && row.dataset.fullText !== undefined) p.textContent = row.dataset.fullText || p.textContent || "";
      row.classList.remove("sync-hidden", "sync-revealed");
      row.dataset.typed = "1";
    });
  }

  function highlightTranscriptAt(time) {
    let activeRow = null;
    document.querySelectorAll(".turn-row[data-start][data-end]").forEach(function (row) {
      const start = parseFloat(row.dataset.start || "0");
      const end = parseFloat(row.dataset.end || "0");
      const active = Number.isFinite(start) && Number.isFinite(end) && end > start && time >= start && time < end;
      row.classList.toggle("active", active);
      if (active) activeRow = row;
    });
    if (activeRow) activeRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function bindAudioTranscriptSync(root) {
    (root || document).querySelectorAll("audio.encounter-audio").forEach(function (audio) {
      if (audio.dataset.syncBound === "1") return;
      audio.dataset.syncBound = "1";
      audio.addEventListener("play", function () {
        activateSyncMode(audio);
      });
      audio.addEventListener("timeupdate", function () {
        const t = audio.currentTime || 0;
        if (!syncMode) activateSyncMode(audio);
        if (syncMode) syncTranscriptAt(t);
        highlightTranscriptAt(t);
      });
      audio.addEventListener("seeked", function () {
        const t = audio.currentTime || 0;
        if (syncMode) syncTranscriptAt(t);
        highlightTranscriptAt(t);
      });
      audio.addEventListener("ended", function () {
        deactivateSyncMode();
      });
      audio.addEventListener("pause", function () {
        if (audio.currentTime === 0 || audio.ended) {
          document.querySelectorAll(".turn-row.active").forEach((row) => row.classList.remove("active"));
        }
      });
    });
  }

  function showSoapLoadingPanel() {
    let panel = document.getElementById("soap-loading-panel");
    if (panel) return;
    panel = document.createElement("section");
    panel.id = "soap-loading-panel";
    panel.className = "soap-loading-panel";
    panel.innerHTML = `
      <div class="soap-loading-body">
        <div class="soap-spinner" aria-hidden="true"></div>
        <div>
          <div class="soap-loading-title">カルテを生成中</div>
          <div class="soap-loading-subtitle">会話・確認事項・画像所見を整理しています。</div>
        </div>
      </div>`;
    document.body.appendChild(panel);
  }

  function resetLivePanels() {
    const transcriptList = document.getElementById("transcript-list");
    if (transcriptList) {
      transcriptList.innerHTML = '<div class="empty-state">Live ASRを開始すると、ここに会話が流れます。</div>';
    }
    const promptsList = document.getElementById("prompts-list");
    if (promptsList) {
      promptsList.innerHTML = '<div class="empty-state">会話から確認事項や注意所見を逐次抽出します。</div>';
    }
    currentTurnText = null;
    currentTurnRow = null;
  }

  function upsertPrompt(data) {
    const list = document.getElementById("prompts-list");
    if (!list) return;
    list.querySelectorAll(".empty-state").forEach((node) => node.remove());
    const key = `${data.kind}:${data.title}`;
    let row = list.querySelector(`[data-prompt-key="${CSS.escape(key)}"]`);
    if (!row) {
      row = document.createElement("article");
      row.dataset.promptKey = key;
      list.appendChild(row);
    }
    row.className = `prompt-row priority-${data.priority || 3}`;
    row.innerHTML = `<strong>${escapeHtml(data.kind_label || data.kind)}: ${escapeHtml(data.title)}</strong><p>${escapeHtml(data.detail)}</p>`;
  }

  document.addEventListener("DOMContentLoaded", function () {
    pollModelStatus();
    bindAudioTranscriptSync(document);
    initializeTranscriptReveal(document);
  });
  document.addEventListener("htmx:afterSwap", function (event) {
    bindAudioTranscriptSync(event.target || document);
    initializeTranscriptReveal(document);
  });
  if (document.readyState !== "loading") {
    pollModelStatus();
    bindAudioTranscriptSync(document);
    initializeTranscriptReveal(document);
  }



  function setLiveButtons(running) {
    const startButton = document.getElementById("start-live-asr");
    const stopButton = document.getElementById("stop-live-asr");
    if (startButton) {
      startButton.disabled = running;
      startButton.textContent = "文字起こし開始";
    }
    if (stopButton) {
      stopButton.disabled = !running;
      stopButton.classList.toggle("btn-danger", running);
      stopButton.classList.toggle("btn-secondary", !running);
      stopButton.textContent = running ? "■ 停止" : "停止";
    }
    const audioCard = document.querySelector(".ws-audio-card");
    if (audioCard) audioCard.classList.toggle("is-recording", running);
  }

  function closeLiveSource() {
    if (liveSource) {
      liveSource.close();
      liveSource = null;
    }
  }

  document.addEventListener("submit", function (event) {
    const form = event.target.closest("form[action*='/soap']");
    if (!form) return;
    const button = form.querySelector("button");
    if (button) {
      button.disabled = true;
      button.textContent = "カルテ作成中...";
    }
    setStatus("カルテを生成しています...", "running");
    showSoapLoadingPanel();
  });

  document.addEventListener("click", function (event) {
    const button = event.target.closest("#start-live-asr");
    if (!button || button.disabled) return;
    const root = document.querySelector("[data-encounter-id]");
    if (!root) return;
    closeLiveSource();
    resetLivePanels();
    setLiveButtons(true);
    setStatus("接続中...", "running");
    const audioEl = document.querySelector("audio.encounter-audio");
    if (audioEl) {
      audioEl.currentTime = 0;
      activateSyncMode(audioEl);
      audioEl.play().catch(function () {});
    }

    liveSource = new EventSource(`/encounters/${root.dataset.encounterId}/stream`);
    liveSource.addEventListener("status", function (event) {
      const data = JSON.parse(event.data);
      setStatus(data.message, data.stage === "done" ? "done" : "running");
      if (data.stage === "done") {
        setLiveButtons(false);
        closeLiveSource();
      }
    });
    liveSource.addEventListener("turn_start", function (event) { beginTurn(JSON.parse(event.data)); });
    liveSource.addEventListener("token", function (event) { appendToken(JSON.parse(event.data)); });
    liveSource.addEventListener("turn_end", function (event) { endTurn(JSON.parse(event.data)); });
    liveSource.addEventListener("prompt", function (event) { upsertPrompt(JSON.parse(event.data)); });
    liveSource.addEventListener("done", function () {
      setLiveButtons(false);
      closeLiveSource();
      trySyncWithPlayingAudio();
    });
    liveSource.addEventListener("stopped", function (event) {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.message || "文字起こしを停止しました。", "done");
      } catch (_) {
        setStatus("文字起こしを停止しました。", "done");
      }
      setLiveButtons(false);
      closeLiveSource();
      trySyncWithPlayingAudio();
    });
    liveSource.addEventListener("error", function (event) {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.message, "error");
      } catch (_) {
        setStatus("SSE接続が終了しました", "error");
      }
      setLiveButtons(false);
      closeLiveSource();
    });
  });
  document.addEventListener("click", async function (event) {
    const button = event.target.closest("#stop-live-asr");
    if (!button || button.disabled) return;
    const root = document.querySelector("[data-encounter-id]");
    if (!root) return;
    button.disabled = true;
    button.textContent = "停止中...";
    setStatus("文字起こしを停止しています...", "running");
    try {
      await fetch(`/encounters/${root.dataset.encounterId}/stream/stop`, { method: "POST" });
    } catch (_) {
      setStatus("停止要求に失敗しました", "error");
      button.disabled = false;
      button.textContent = "停止";
    }
  });

  // Audio filename display — works after htmx swaps via event delegation
  document.addEventListener("change", function (event) {
    var inp = event.target;
    if (inp.type !== "file" || !inp.closest("#audio-panel")) return;
    var lbl = inp.closest("form").querySelector(".ws-audio-filename");
    if (lbl && inp.files[0]) lbl.textContent = inp.files[0].name;
  });

})();
