/**
 * English to Hindi Machine Translation System - Frontend Client
 * Connects directly to the FastAPI Neural Machine Translation backend.
 */

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';
const TRANSLATE_ENDPOINT = `${API_BASE_URL}/translate`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;
const MODEL_INFO_ENDPOINT = `${API_BASE_URL}/model-info`;

// DOM Elements
const englishInput = document.getElementById('englishInput');
const charCount = document.getElementById('charCount');
const btnClearInput = document.getElementById('btnClearInput');
const btnPasteText = document.getElementById('btnPasteText');

const translationOutput = document.getElementById('translationOutput');
const loadingOverlay = document.getElementById('loadingOverlay');
const outputWordCount = document.getElementById('outputWordCount');
const latencyBadge = document.getElementById('latencyBadge');
const btnCopyTop = document.getElementById('btnCopyTop');
const btnCopyTranslation = document.getElementById('btnCopyTranslation');
const copyBtnText = document.getElementById('copyBtnText');

const btnTranslate = document.getElementById('btnTranslate');
const translateBtnText = document.getElementById('translateBtnText');
const translateBtnIcon = document.getElementById('translateBtnIcon');
const btnClear = document.getElementById('btnClear');

const statusAlert = document.getElementById('statusAlert');
const alertMessage = document.getElementById('alertMessage');
const btnCloseAlert = document.getElementById('btnCloseAlert');

const backendStatusIndicator = document.getElementById('backendStatusIndicator');
const backendStatusText = document.getElementById('backendStatusText');
const toastContainer = document.getElementById('toastContainer');
const exampleChips = document.querySelectorAll('.example-chip');

// State
let isTranslating = false;
let currentHindiTranslation = '';

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setupEventListeners();
  updateCharCount();
});

// ==========================================================================
// Event Listeners
// ==========================================================================
function setupEventListeners() {
  // Input typing and counter
  englishInput.addEventListener('input', () => {
    updateCharCount();
    hideAlert();
  });

  // Keyboard shortcut: Ctrl+Enter or Cmd+Enter to translate
  englishInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleTranslate();
    }
  });

  // Action Buttons
  btnTranslate.addEventListener('click', handleTranslate);
  btnClear.addEventListener('click', handleClearAll);
  btnClearInput.addEventListener('click', handleClearInput);
  btnCopyTranslation.addEventListener('click', handleCopy);
  btnCopyTop.addEventListener('click', handleCopy);
  btnCloseAlert.addEventListener('click', hideAlert);

  // Paste shortcut button
  btnPasteText.addEventListener('click', async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        englishInput.value = text.slice(0, 500);
        updateCharCount();
        englishInput.focus();
        showToast('Text pasted from clipboard', 'info');
      }
    } catch {
      englishInput.focus();
    }
  });

  // Example Sentences
  exampleChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const sentence = chip.getAttribute('data-text');
      if (sentence) {
        englishInput.value = sentence;
        updateCharCount();
        resetOutputPlaceholder();
        hideAlert();
        englishInput.focus();
        showToast(`Loaded example: "${sentence}"`, 'info');
      }
    });
  });

  // Re-check backend health periodically (every 30s)
  setInterval(checkBackendHealth, 30000);
}

// ==========================================================================
// Backend Health Check
// ==========================================================================
async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(HEALTH_ENDPOINT, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.status === 'ok') {
        setBackendStatus(true, 'API Connected');
        return;
      }
    }
    setBackendStatus(false, 'API Degraded');
  } catch {
    setBackendStatus(false, 'API Offline');
  }
}

function setBackendStatus(isOnline, text) {
  backendStatusIndicator.classList.remove('checking', 'connected', 'error');
  if (isOnline) {
    backendStatusIndicator.classList.add('connected');
  } else {
    backendStatusIndicator.classList.add('error');
  }
  backendStatusText.textContent = text;
}

// ==========================================================================
// Translation Execution
// ==========================================================================
async function handleTranslate() {
  if (isTranslating) return;

  const rawText = englishInput.value.trim();

  // Validate input
  if (!rawText) {
    showAlert('Please enter an English sentence to translate.');
    englishInput.focus();
    return;
  }

  hideAlert();
  setLoadingState(true);
  const startTime = performance.now();

  try {
    const response = await fetch(TRANSLATE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ text: rawText }),
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok) {
      let errorMsg = 'Translation service returned an error.';
      try {
        const errData = await response.json();
        if (errData && errData.detail) {
          errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
        }
      } catch {
        // use default error message
      }
      throw new Error(errorMsg);
    }

    const data = await response.json();
    const hindiResult = data.translation || '';

    // Render result
    displayTranslation(hindiResult, elapsed);
    setBackendStatus(true, 'API Connected');
  } catch (err) {
    let friendlyMessage = 'Unable to translate text.';

    if (err.name === 'AbortError' || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      friendlyMessage = 'Cannot connect to the FastAPI translation server at http://127.0.0.1:8000. Please ensure the backend is started.';
      setBackendStatus(false, 'API Offline');
    } else {
      friendlyMessage = `Translation notice: ${err.message}`;
    }

    showAlert(friendlyMessage);
    resetOutputPlaceholder();
  } finally {
    setLoadingState(false);
  }
}

// ==========================================================================
// UI State Updates
// ==========================================================================
function displayTranslation(hindiText, elapsedMs) {
  currentHindiTranslation = hindiText;
  translationOutput.classList.remove('placeholder-state');
  translationOutput.textContent = hindiText;

  // Update words count
  const words = hindiText.trim() ? hindiText.trim().split(/\s+/).length : 0;
  outputWordCount.textContent = `${words} word${words === 1 ? '' : 's'}`;

  // Update latency badge
  if (elapsedMs !== undefined) {
    latencyBadge.textContent = `⚡ ${elapsedMs}ms`;
    latencyBadge.classList.remove('hidden');
  }

  // Enable copy buttons
  btnCopyTranslation.disabled = !hindiText;
  btnCopyTop.disabled = !hindiText;
}

function resetOutputPlaceholder() {
  currentHindiTranslation = '';
  translationOutput.classList.add('placeholder-state');
  translationOutput.textContent = 'Your Hindi translation will appear here...';
  outputWordCount.textContent = '0 words';
  latencyBadge.classList.add('hidden');
  btnCopyTranslation.disabled = true;
  btnCopyTop.disabled = true;
}

function setLoadingState(loading) {
  isTranslating = loading;
  btnTranslate.disabled = loading;

  if (loading) {
    translateBtnText.textContent = 'Translating...';
    translateBtnIcon.textContent = '⏳';
    loadingOverlay.classList.remove('hidden');
  } else {
    translateBtnText.textContent = 'Translate';
    translateBtnIcon.textContent = '⚡';
    loadingOverlay.classList.add('hidden');
  }
}

function updateCharCount() {
  const length = englishInput.value.length;
  charCount.textContent = length;
}

function handleClearAll() {
  englishInput.value = '';
  updateCharCount();
  resetOutputPlaceholder();
  hideAlert();
  englishInput.focus();
}

function handleClearInput() {
  englishInput.value = '';
  updateCharCount();
  hideAlert();
  englishInput.focus();
}

// ==========================================================================
// Copy to Clipboard
// ==========================================================================
async function handleCopy() {
  if (!currentHindiTranslation) return;

  try {
    await navigator.clipboard.writeText(currentHindiTranslation);

    // Visual feedback on button
    btnCopyTranslation.classList.add('copied');
    copyBtnText.textContent = 'Copied!';
    btnCopyTop.textContent = '✓';

    showToast('Hindi translation copied to clipboard!', 'success');

    setTimeout(() => {
      btnCopyTranslation.classList.remove('copied');
      copyBtnText.textContent = 'Copy Translation';
      btnCopyTop.textContent = '📋';
    }, 2000);
  } catch {
    showToast('Could not copy to clipboard. Please copy manually.', 'error');
  }
}

// ==========================================================================
// Alerts & Toasts
// ==========================================================================
function showAlert(msg) {
  alertMessage.textContent = msg;
  statusAlert.classList.remove('hidden');
}

function hideAlert() {
  statusAlert.classList.add('hidden');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type === 'success' ? 'toast-success' : ''}`;
  
  const icon = type === 'success' ? '✓' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;

  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'all 0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, 2500);
}
