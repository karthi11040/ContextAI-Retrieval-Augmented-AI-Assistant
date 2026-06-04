/**
 * ContextAI RAG — Enhanced Frontend
 * Features: themes, shortcuts, toasts, drag-drop, search, streaming, confetti
 */

// ============================================
// GLOBAL STATE
// ============================================
const State = {
    apiKey: '',
    embeddingModel: 'openai/text-embedding-3-small',
    generationModel: 'google/gemini-2.5-flash',
    temperature: 0.5,
    topK: 5,
    hybridWeight: 0.5,
    systemPrompt: '',

    indexed: false,
    chatHistory: [],
    conversations: {},
    currentSessionId: null,
    isIndexingActive: false,

    // New features
    theme: 'dark',
    sidebarCollapsed: false,
    streamingEnabled: true,
    autoCitations: true,
    soundEnabled: false,
    confettiEnabled: true,

    // Chat search
    chatSearchQuery: '',
    chatSearchMatches: [],
    chatSearchIndex: -1,

    // Context menu
    ctxMenuTarget: null,

    // Upload queue
    uploadQueue: [],

    // Explorer filter
    explorerFilter: 'all'
};

// ============================================
// DOM REFERENCES
// ============================================
const DOM = {
    sidebar: document.getElementById('sidebar-nav'),
    sidebarToggleBtn: document.getElementById('sidebar-toggle-btn'),
    navTabs: document.getElementById('navigation-tabs'),
    tabBtns: document.querySelectorAll('.nav-btn'),
    viewPanels: document.querySelectorAll('.view-panel'),
    globalStatusBadge: document.getElementById('global-status-badge'),
    globalStatusText: document.getElementById('global-status-text'),
    workspaceTitle: document.getElementById('workspace-title'),
    workspaceTag: document.getElementById('workspace-tag'),
    activeModelBadge: document.getElementById('active-model-badge'),

    globalIndexingPanel: document.getElementById('global-indexing-panel'),
    alertTitleText: document.getElementById('alert-title-text'),
    alertProgressFill: document.getElementById('alert-progress-fill'),
    alertProgressPercent: document.getElementById('alert-progress-percentage'),

    btnNewChat: document.getElementById('btn-new-chat'),
    recentChatsList: document.getElementById('recent-chats-list'),
    chatSearchInput: document.getElementById('chat-search-input'),
    btnExportChat: document.getElementById('btn-export-chat'),
    headerMenu: document.getElementById('header-menu'),

    chatScroller: document.getElementById('chat-history-container'),
    chatForm: document.getElementById('chat-input-form'),
    chatInput: document.getElementById('chat-input-textarea'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    voiceBtn: document.getElementById('voice-btn'),

    idxStatPdf: document.getElementById('idx-stat-pdf'),
    idxStatIndexed: document.getElementById('idx-stat-indexed'),
    idxStatPages: document.getElementById('idx-stat-pages'),
    idxStatChunks: document.getElementById('idx-stat-chunks'),
    idxStatFaiss: document.getElementById('idx-stat-faiss'),
    idxStatVocab: document.getElementById('idx-stat-vocab'),
    indexerConfigForm: document.getElementById('indexer-config-form'),
    configChunkSize: document.getElementById('config-chunk-size'),
    configChunkOverlap: document.getElementById('config-chunk-overlap'),
    startIndexingBtn: document.getElementById('start-indexing-btn'),

    dropZone: document.getElementById('drop-zone'),
    docUploadInput: document.getElementById('document-upload-input'),
    uploadDocBtn: document.getElementById('upload-doc-btn'),
    uploadFileList: document.getElementById('upload-file-list'),
    uploadStatusMsg: document.getElementById('upload-status-message'),
    dropZoneText: document.getElementById('drop-zone-text'),
    chatAttachBtn: document.getElementById('chat-attach-btn'),

    explorerSearchForm: document.getElementById('explorer-search-form'),
    explorerSearchInput: document.getElementById('explorer-search-input'),
    explorerResultsGrid: document.getElementById('explorer-results-grid'),
    resultsCountBanner: document.getElementById('results-count-banner'),
    resultsCountText: document.getElementById('results-count-text'),

    settingsForm: document.getElementById('settings-config-form'),
    settingsApiKey: document.getElementById('settings-api-key'),
    settingsEmbedModel: document.getElementById('settings-embed-model'),
    settingsGenModel: document.getElementById('settings-gen-model'),
    settingsSystemPrompt: document.getElementById('settings-system-prompt'),
    settingsTopK: document.getElementById('settings-top-k'),
    lblTopK: document.getElementById('lbl-top-k'),
    settingsTemperature: document.getElementById('settings-temperature'),
    lblTemperature: document.getElementById('lbl-temperature'),
    settingsHybridWeight: document.getElementById('settings-hybrid-weight'),
    lblHybridWeight: document.getElementById('lbl-hybrid-weight'),
    toggleKeyVisibility: document.getElementById('toggle-key-visibility'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),

    toastContainer: document.getElementById('toast-container'),
    shortcutsModal: document.getElementById('shortcuts-modal'),
    contextMenu: document.getElementById('context-menu'),
    pagePreviewPopup: document.getElementById('page-preview-popup'),
    pagePreviewContent: document.getElementById('page-preview-content'),
    chatSearchOverlay: document.getElementById('chat-search-overlay'),
    chatSearchField: document.getElementById('chat-search-field'),
    chatSearchCount: document.getElementById('chat-search-count'),
    mobileOverlay: document.getElementById('mobile-overlay')
};

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    loadLocalSettings();
    loadConversationsDB();
    bindEvents();
    initDragDrop();
    initKeyboardShortcuts();
    initContextMenu();
    checkServerStatus(true);
    startStatusPolling();
    applyTheme(State.theme);

    // Auto-focus input on load
    setTimeout(() => DOM.chatInput.focus(), 300);
});

// ============================================
// THEME SYSTEM
// ============================================
function setTheme(theme) {
    State.theme = theme;
    applyTheme(theme);
    localStorage.setItem('rag_theme', theme);

    // Update active dot
    document.querySelectorAll('.theme-dot').forEach((dot, i) => {
        const themes = ['dark', 'light', 'midnight'];
        dot.classList.toggle('active', themes[i] === theme);
    });

    // Re-highlight code if any
    document.querySelectorAll('pre code').forEach(block => {
        if (window.hljs) hljs.highlightElement(block);
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);

    // Update highlight.js theme
    const hlLink = document.querySelector('link[href*="highlight.js"]');
    if (hlLink) {
        const isDark = theme !== 'light';
        hlLink.href = isDark 
            ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css'
            : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css';
    }
}

// ============================================
// SIDEBAR
// ============================================
function toggleSidebar() {
    State.sidebarCollapsed = !State.sidebarCollapsed;
    DOM.sidebar.classList.toggle('collapsed', State.sidebarCollapsed);
    const icon = DOM.sidebarToggleBtn.querySelector('i');
    icon.className = State.sidebarCollapsed ? 'fa-solid fa-chevron-right text-xs' : 'fa-solid fa-chevron-left text-xs';
    DOM.sidebarToggleBtn.setAttribute('data-tooltip', State.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
}

function toggleMobileSidebar() {
    DOM.sidebar.classList.toggle('mobile-open');
    DOM.mobileOverlay.classList.toggle('active');
}

// ============================================
// TOAST NOTIFICATIONS
// ============================================
function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = 'toast';

    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-exclamation',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info'
    };

    const colors = {
        success: 'var(--accent-emerald)',
        error: 'var(--accent-rose)',
        warning: 'var(--accent-amber)',
        info: 'var(--accent-sky)'
    };

    toast.innerHTML = `
        <i class="fa-solid ${icons[type]} text-lg shrink-0" style="color: ${colors[type]}; margin-top: 2px;"></i>
        <div class="flex-grow">
            <p class="text-sm font-medium" style="color: var(--text-primary);">${message}</p>
        </div>
        <button class="btn-ghost p-1 rounded shrink-0" onclick="this.parentElement.remove()">
            <i class="fa-solid fa-xmark text-xs"></i>
        </button>
    `;

    DOM.toastContainer.appendChild(toast);

    if (duration > 0) {
        setTimeout(() => {
            toast.classList.add('toast-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    return toast;
}

// ============================================
// SETTINGS
// ============================================
function loadLocalSettings() {
    State.apiKey = localStorage.getItem('rag_api_key') || '';
    State.embeddingModel = localStorage.getItem('rag_embed_model') || 'openai/text-embedding-3-small';
    State.generationModel = localStorage.getItem('rag_gen_model') || 'google/gemini-2.5-flash';
    State.temperature = parseFloat(localStorage.getItem('rag_temperature')) || 0.5;
    State.topK = parseInt(localStorage.getItem('rag_top_k')) || 5;
    State.hybridWeight = parseFloat(localStorage.getItem('rag_hybrid_weight')) || 0.5;
    State.systemPrompt = localStorage.getItem('rag_system_prompt') || '';
    State.theme = localStorage.getItem('rag_theme') || 'dark';
    State.streamingEnabled = localStorage.getItem('rag_streaming') !== 'false';
    State.autoCitations = localStorage.getItem('rag_citations') !== 'false';
    State.soundEnabled = localStorage.getItem('rag_sound') === 'true';
    State.confettiEnabled = localStorage.getItem('rag_confetti') !== 'false';

    DOM.settingsApiKey.value = State.apiKey;
    DOM.settingsEmbedModel.value = State.embeddingModel;
    DOM.settingsGenModel.value = State.generationModel;
    DOM.settingsSystemPrompt.value = State.systemPrompt || getDefaultSystemPrompt();

    DOM.settingsTopK.value = State.topK;
    DOM.lblTopK.textContent = State.topK;

    DOM.settingsTemperature.value = State.temperature;
    DOM.lblTemperature.textContent = State.temperature.toFixed(1);

    DOM.settingsHybridWeight.value = State.hybridWeight;
    DOM.lblHybridWeight.textContent = State.hybridWeight.toFixed(1);

    // Toggle states
    document.getElementById('toggle-streaming').classList.toggle('active', State.streamingEnabled);
    document.getElementById('toggle-citations').classList.toggle('active', State.autoCitations);
    document.getElementById('toggle-sound').classList.toggle('active', State.soundEnabled);
    document.getElementById('toggle-confetti').classList.toggle('active', State.confettiEnabled);

    updateHeaderBadge();
    applyTheme(State.theme);

    // Set active theme dot
    const themes = ['dark', 'light', 'midnight'];
    document.querySelectorAll('.theme-dot').forEach((dot, i) => {
        dot.classList.toggle('active', themes[i] === State.theme);
    });
}

function getDefaultSystemPrompt() {
    return `You are an expert AI teaching assistant with deep knowledge of Stuart Russell and Peter Norvig's 'Artificial Intelligence: A Modern Approach'. Provide accurate, well-cited academic answers.`;
}

function saveLocalSettings() {
    localStorage.setItem('rag_api_key', State.apiKey);
    localStorage.setItem('rag_embed_model', State.embeddingModel);
    localStorage.setItem('rag_gen_model', State.generationModel);
    localStorage.setItem('rag_top_k', State.topK);
    localStorage.setItem('rag_temperature', State.temperature);
    localStorage.setItem('rag_hybrid_weight', State.hybridWeight);
    localStorage.setItem('rag_system_prompt', State.systemPrompt);
    localStorage.setItem('rag_theme', State.theme);
    localStorage.setItem('rag_streaming', State.streamingEnabled);
    localStorage.setItem('rag_citations', State.autoCitations);
    localStorage.setItem('rag_sound', State.soundEnabled);
    localStorage.setItem('rag_confetti', State.confettiEnabled);

    updateHeaderBadge();
}

function updateHeaderBadge() {
    const shortName = State.generationModel.split('/').pop();
    DOM.activeModelBadge.textContent = shortName;
}

function resetSystemPrompt() {
    DOM.settingsSystemPrompt.value = getDefaultSystemPrompt();
}

function resetAllSettings() {
    if (!confirm('Reset all settings to default? This will clear your API key and preferences.')) return;

    localStorage.removeItem('rag_api_key');
    localStorage.removeItem('rag_embed_model');
    localStorage.removeItem('rag_gen_model');
    localStorage.removeItem('rag_temperature');
    localStorage.removeItem('rag_top_k');
    localStorage.removeItem('rag_hybrid_weight');
    localStorage.removeItem('rag_system_prompt');
    localStorage.removeItem('rag_theme');
    localStorage.removeItem('rag_streaming');
    localStorage.removeItem('rag_citations');
    localStorage.removeItem('rag_sound');
    localStorage.removeItem('rag_confetti');

    loadLocalSettings();
    showToast('All settings reset to default', 'success');
}

// ============================================
// CONVERSATIONS DATABASE
// ============================================
function loadConversationsDB() {
    const raw = localStorage.getItem('rag_conversations');
    if (raw) {
        try { State.conversations = JSON.parse(raw); } 
        catch (e) { State.conversations = {}; }
    } else { State.conversations = {}; }

    renderConversationsList();

    const sessionIds = Object.keys(State.conversations);
    if (sessionIds.length > 0) {
        sessionIds.sort((a, b) => State.conversations[b].timestamp - State.conversations[a].timestamp);
        loadConversation(sessionIds[0]);
    } else { startNewConversation(); }
}

function renderConversationsList() {
    DOM.recentChatsList.innerHTML = '';
    const sessionIds = Object.keys(State.conversations);

    if (sessionIds.length === 0) {
        DOM.recentChatsList.innerHTML = `<div class="text-xs py-3 px-3 text-center" style="color: var(--text-tertiary);">No conversations saved</div>`;
        return;
    }

    sessionIds.sort((a, b) => State.conversations[b].timestamp - State.conversations[a].timestamp);

    sessionIds.forEach(id => {
        const item = State.conversations[id];
        const isActive = id === State.currentSessionId;

        const chatRow = document.createElement('button');
        chatRow.className = `group text-left rounded-xl text-sm leading-5 flex p-2.5 items-center justify-between gap-2 w-full transition cursor-pointer select-none ${
            isActive ? 'font-medium' : ''
        }`;
        chatRow.style.cssText = isActive 
            ? `background: var(--bg-tertiary); color: var(--text-primary);` 
            : `color: var(--text-tertiary);`;

        if (!isActive) {
            chatRow.addEventListener('mouseenter', () => {
                chatRow.style.background = 'var(--bg-tertiary)';
                chatRow.style.color = 'var(--text-primary)';
            });
            chatRow.addEventListener('mouseleave', () => {
                chatRow.style.background = 'transparent';
                chatRow.style.color = 'var(--text-tertiary)';
            });
        }

        chatRow.addEventListener('click', () => loadConversation(id));

        const timeStr = formatTimeAgo(item.timestamp);

        chatRow.innerHTML = `
            <div class="truncate flex-grow flex items-center gap-2.5 mr-2 min-w-0">
                <i class="fa-solid fa-message text-xs w-4 text-center shrink-0" style="color: ${isActive ? 'var(--accent-sky)' : 'inherit'};"></i>
                <div class="flex flex-col min-w-0">
                    <span class="truncate">${escapeHtml(item.title)}</span>
                    <span class="text-[10px] truncate" style="color: var(--text-tertiary);">${timeStr} · ${item.messages.length} msgs</span>
                </div>
            </div>
            <span class="opacity-0 group-hover:opacity-100 p-1 rounded transition cursor-pointer delete-chat-btn shrink-0" 
                style="color: var(--text-tertiary);" title="Delete">
                <i class="fa-solid fa-trash-can text-[10px]"></i>
            </span>
        `;

        const delBtn = chatRow.querySelector('.delete-chat-btn');
        delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteConversation(id);
        });

        DOM.recentChatsList.appendChild(chatRow);
    });
}

function filterConversations(query) {
    const items = DOM.recentChatsList.querySelectorAll('button');
    const lower = query.toLowerCase();
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(lower) ? '' : 'none';
    });
}

function formatTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(timestamp).toLocaleDateString();
}

function startNewConversation() {
    State.currentSessionId = `session-${Date.now()}`;
    State.chatHistory = [];

    DOM.chatScroller.innerHTML = `
        <div class="assistant-welcome flex justify-center items-center h-full py-10">
            <div class="card rounded-2xl p-8 text-center max-w-lg shadow-xl animate-slideIn" style="background: var(--glass-bg); backdrop-filter: blur(20px);">
                <div class="size-16 rounded-2xl mx-auto mb-5 flex items-center justify-center animate-float" style="background: linear-gradient(135deg, var(--accent-purple), var(--accent-sky));">
                    <i class="fa-solid fa-graduation-cap text-2xl text-white"></i>
                </div>
                <h2 class="text-lg font-bold mb-2 font-heading" style="color: var(--text-primary);">AI Teaching Assistant</h2>
                <p class="text-sm mb-4 leading-relaxed font-medium" style="color: var(--text-secondary);">Connected to <strong>Artificial Intelligence: A Modern Approach (3rd ed.)</strong></p>
                <p class="p-4 rounded-xl text-xs leading-relaxed mb-6" style="background: var(--bg-tertiary); color: var(--text-tertiary); border: 1px solid var(--border-color);">
                    Ask academic questions, explain algorithms, or query specific concepts. Answers include precise page citations.
                </p>
                <div class="flex flex-wrap justify-center gap-2 mb-6">
                    <button class="quick-action" data-query="Explain what A* Search is and its heuristic requirements.">A* Search & Heuristics</button>
                    <button class="quick-action" data-query="What does PEAS stand for in Artificial Intelligence? Provide examples.">What is PEAS?</button>
                    <button class="quick-action" data-query="Define Markov Decision Processes (MDPs) and the Bellman Equation.">MDPs & Bellman</button>
                    <button class="quick-action" data-query="Explain the difference between supervised and unsupervised learning.">Supervised vs Unsupervised</button>
                    <button class="quick-action" data-query="What is the Turing Test and its criticisms?">Turing Test</button>
                </div>
                <div class="flex items-center justify-center gap-4 text-[10px]" style="color: var(--text-tertiary);">
                    <span class="flex items-center gap-1"><i class="fa-solid fa-bolt text-[8px]" style="color: var(--accent-amber);"></i> Real-time retrieval</span>
                    <span class="flex items-center gap-1"><i class="fa-solid fa-book text-[8px]" style="color: var(--accent-sky);"></i> Page citations</span>
                    <span class="flex items-center gap-1"><i class="fa-solid fa-shield-halved text-[8px]" style="color: var(--accent-emerald);"></i> Academic verified</span>
                </div>
            </div>
        </div>
    `;

    bindWelcomeChips();
    DOM.workspaceTitle.textContent = "New Conversation";
    DOM.workspaceTag.classList.add('hidden');
    DOM.btnExportChat.classList.add('hidden');
    renderConversationsList();
}

function bindWelcomeChips() {
    DOM.chatScroller.querySelectorAll('.quick-action').forEach(chip => {
        chip.addEventListener('click', () => {
            DOM.chatInput.value = chip.dataset.query;
            DOM.chatInput.focus();
            DOM.chatForm.requestSubmit();
        });
    });
}

function loadConversation(id) {
    if (!State.conversations[id]) return;

    State.currentSessionId = id;
    const session = State.conversations[id];
    State.chatHistory = session.messages;

    DOM.chatScroller.innerHTML = '';
    DOM.workspaceTitle.textContent = session.title;
    DOM.workspaceTag.classList.remove('hidden');
    DOM.workspaceTag.textContent = session.tag || 'Academic';

    if (State.chatHistory.length === 0) { startNewConversation(); return; }

    State.chatHistory.forEach(msg => {
        appendMessage(msg.role, msg.content, msg.metadata, false);
    });

    DOM.btnExportChat.classList.remove('hidden');
    scrollToBottom();
    renderConversationsList();
}

function saveActiveConversation() {
    if (State.chatHistory.length === 0) return;

    let title = "New Conversation";
    const firstUserQuery = State.chatHistory.find(m => m.role === 'user');
    if (firstUserQuery) {
        title = firstUserQuery.content;
        if (title.length > 30) title = title.substring(0, 28) + "...";
    }

    State.conversations[State.currentSessionId] = {
        id: State.currentSessionId,
        title: title,
        tag: 'Academic',
        messages: State.chatHistory,
        timestamp: Date.now()
    };

    localStorage.setItem('rag_conversations', JSON.stringify(State.conversations));
    DOM.workspaceTitle.textContent = title;
    DOM.workspaceTag.classList.remove('hidden');
    DOM.workspaceTag.textContent = 'Academic';
    DOM.btnExportChat.classList.remove('hidden');
    renderConversationsList();
}

function deleteConversation(id) {
    if (!confirm("Delete this conversation permanently?")) return;
    delete State.conversations[id];
    localStorage.setItem('rag_conversations', JSON.stringify(State.conversations));
    if (State.currentSessionId === id) startNewConversation();
    else renderConversationsList();
    showToast('Conversation deleted', 'success');
}

function clearAllConversations() {
    if (!confirm("Delete ALL conversations? This cannot be undone.")) return;
    State.conversations = {};
    localStorage.removeItem('rag_conversations');
    startNewConversation();
    showToast('All conversations cleared', 'success');
}

function clearCurrentChat() {
    if (!confirm("Clear current chat?")) return;
    State.chatHistory = [];
    saveActiveConversation();
    startNewConversation();
    toggleHeaderMenu();
}

function renameConversation() {
    const session = State.conversations[State.currentSessionId];
    if (!session) return;
    const newName = prompt("Rename conversation:", session.title);
    if (newName && newName.trim()) {
        session.title = newName.trim();
        localStorage.setItem('rag_conversations', JSON.stringify(State.conversations));
        DOM.workspaceTitle.textContent = session.title;
        renderConversationsList();
        showToast('Renamed successfully', 'success');
    }
    toggleHeaderMenu();
}

function duplicateConversation() {
    const session = State.conversations[State.currentSessionId];
    if (!session) return;
    const newId = `session-${Date.now()}`;
    State.conversations[newId] = {
        ...session,
        id: newId,
        title: session.title + ' (Copy)',
        timestamp: Date.now()
    };
    localStorage.setItem('rag_conversations', JSON.stringify(State.conversations));
    loadConversation(newId);
    showToast('Conversation duplicated', 'success');
    toggleHeaderMenu();
}

function exportChatLog() {
    if (State.chatHistory.length === 0) return;

    const formats = ['markdown', 'json', 'txt'];
    const format = prompt("Export format: markdown, json, or txt?", "markdown");
    if (!format || !formats.includes(format.toLowerCase())) return;

    let content, mime, ext;
    const title = DOM.workspaceTitle.textContent;

    if (format === 'markdown') {
        let md = `# ${title}\n\n`;
        md += `*Exported: ${new Date().toLocaleString()}*\n\n---\n\n`;
        State.chatHistory.forEach(msg => {
            md += msg.role === 'user' ? `## User\n\n${msg.content}\n\n` : `## Assistant\n\n${msg.content}\n\n`;
            if (msg.metadata?.retrieved_chunks?.length) {
                md += `**Sources:**\n`;
                msg.metadata.retrieved_chunks.forEach(c => {
                    md += `- Page ${c.page_start} (${Math.round(c.similarity_score * 100)}% match)\n`;
                });
                md += `\n`;
            }
            md += `---\n\n`;
        });
        content = md; mime = 'text/markdown'; ext = 'md';
    } else if (format === 'json') {
        content = JSON.stringify({
            title, exported: new Date().toISOString(), messages: State.chatHistory
        }, null, 2);
        mime = 'application/json'; ext = 'json';
    } else {
        let txt = `${title}\n${'='.repeat(title.length)}\n\n`;
        State.chatHistory.forEach(msg => {
            txt += `[${msg.role.toUpperCase()}]\n${msg.content}\n\n`;
        });
        content = txt; mime = 'text/plain'; ext = 'txt';
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `rag_chat_${title.toLowerCase().replace(/[^a-z0-9]+/g, '_') || 'log'}.${ext}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showToast(`Exported as ${format}`, 'success');
}

// ============================================
// EVENT BINDING
// ============================================
function bindEvents() {
    DOM.btnNewChat.addEventListener('click', startNewConversation);
    DOM.btnExportChat.addEventListener('click', exportChatLog);

    DOM.navTabs.addEventListener('click', (e) => {
        const btn = e.target.closest('.nav-btn');
        if (!btn) return;
        switchTab(btn.dataset.tab, btn);
    });

    DOM.toggleKeyVisibility.addEventListener('click', () => {
        const input = DOM.settingsApiKey;
        const icon = DOM.toggleKeyVisibility.querySelector('i');
        input.type = input.type === 'password' ? 'text' : 'password';
        icon.className = input.type === 'password' ? 'fa-solid fa-eye-slash text-sm' : 'fa-solid fa-eye text-sm';
    });

    DOM.settingsTopK.addEventListener('input', (e) => { DOM.lblTopK.textContent = e.target.value; });
    DOM.settingsTemperature.addEventListener('input', (e) => { DOM.lblTemperature.textContent = parseFloat(e.target.value).toFixed(1); });
    DOM.settingsHybridWeight.addEventListener('input', (e) => { DOM.lblHybridWeight.textContent = parseFloat(e.target.value).toFixed(1); });

    DOM.settingsForm.addEventListener('submit', handleSettingsSubmit);
    DOM.indexerConfigForm.addEventListener('submit', handleIndexingSubmit);
    DOM.explorerSearchForm.addEventListener('submit', handleExplorerSearchSubmit);
    DOM.chatForm.addEventListener('submit', handleChatSubmit);

    DOM.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            DOM.chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
    });

    DOM.chatInput.addEventListener('input', autoResizeTextarea);

    DOM.docUploadInput.addEventListener('change', handleFileUploadChange);
    DOM.uploadDocBtn.addEventListener('click', handleDocUploadSubmit);
    DOM.chatAttachBtn.addEventListener('click', () => {
        const indexerTab = document.getElementById('tab-btn-indexer');
        if (indexerTab) switchTab('indexer-view', indexerTab);
    });

    // Explorer filters
    document.querySelectorAll('[data-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active-filter'));
            btn.classList.add('active-filter');
            State.explorerFilter = btn.dataset.filter;
        });
    });

    // Voice button (placeholder)
    DOM.voiceBtn.addEventListener('click', () => {
        showToast('Voice input coming soon!', 'info');
    });

    // Click outside to close menus
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#header-menu') && !e.target.closest('[onclick="toggleHeaderMenu()"]')) {
            DOM.headerMenu.classList.add('hidden');
        }
    });
}

function autoResizeTextarea() {
    DOM.chatInput.style.height = 'auto';
    DOM.chatInput.style.height = Math.min(DOM.chatInput.scrollHeight, 200) + 'px';
}

function toggleHeaderMenu() {
    DOM.headerMenu.classList.toggle('hidden');
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            if (e.key === 'Escape') {
                if (!DOM.shortcutsModal.classList.contains('hidden')) hideShortcutsModal();
                if (!DOM.chatSearchOverlay.classList.contains('active')) DOM.chatInput.blur();
            }
            return;
        }

        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case 'n': e.preventDefault(); startNewConversation(); break;
                case 'f': e.preventDefault(); toggleChatSearch(); break;
                case 'b': e.preventDefault(); toggleSidebar(); break;
                case '/': e.preventDefault(); DOM.chatInput.focus(); break;
                case 'e': e.preventDefault(); exportChatLog(); break;
                case 'k': e.preventDefault(); showShortcutsModal(); break;
            }
        }

        if (e.key === 'Escape') {
            hideShortcutsModal();
            DOM.chatSearchOverlay.classList.remove('active');
            DOM.contextMenu.classList.remove('active');
        }
    });
}

function showShortcutsModal() {
    DOM.shortcutsModal.classList.remove('hidden');
    DOM.shortcutsModal.classList.add('flex');
}

function hideShortcutsModal() {
    DOM.shortcutsModal.classList.add('hidden');
    DOM.shortcutsModal.classList.remove('flex');
}

// ============================================
// CONTEXT MENU
// ============================================
function initContextMenu() {
    DOM.chatScroller.addEventListener('contextmenu', (e) => {
        const msgRow = e.target.closest('.message-row');
        if (!msgRow) return;
        e.preventDefault();
        State.ctxMenuTarget = msgRow;

        DOM.contextMenu.style.left = `${e.clientX}px`;
        DOM.contextMenu.style.top = `${e.clientY}px`;
        DOM.contextMenu.classList.add('active');
    });

    document.addEventListener('click', () => {
        DOM.contextMenu.classList.remove('active');
    });
}

function ctxCopyMessage() {
    if (!State.ctxMenuTarget) return;
    const content = State.ctxMenuTarget.querySelector('.msg-text-content');
    if (content) {
        navigator.clipboard.writeText(content.innerText);
        showToast('Copied to clipboard', 'success');
    }
}

function ctxQuoteMessage() {
    if (!State.ctxMenuTarget) return;
    const content = State.ctxMenuTarget.querySelector('.msg-text-content');
    if (content) {
        DOM.chatInput.value = `> ${content.innerText.substring(0, 200)}\n\n`;
        DOM.chatInput.focus();
        autoResizeTextarea();
    }
}

function ctxRegenerateMessage() {
    if (!State.ctxMenuTarget) return;
    // Find last user message before this and resubmit
    const allRows = [...DOM.chatScroller.querySelectorAll('.message-row')];
    const idx = allRows.indexOf(State.ctxMenuTarget);
    if (idx > 0) {
        const prevUser = allRows.slice(0, idx).reverse().find(r => r.dataset.role === 'user');
        if (prevUser) {
            const query = prevUser.querySelector('.msg-text-content')?.innerText || '';
            DOM.chatInput.value = query;
            DOM.chatForm.requestSubmit();
        }
    }
}

function ctxDeleteMessage() {
    if (!State.ctxMenuTarget) return;
    if (!confirm('Delete this message?')) return;
    State.ctxMenuTarget.remove();
    // Rebuild history from DOM
    rebuildHistoryFromDOM();
}

function rebuildHistoryFromDOM() {
    State.chatHistory = [];
    DOM.chatScroller.querySelectorAll('.message-row').forEach(row => {
        const role = row.dataset.role;
        const content = row.querySelector('.msg-text-content')?.innerText || '';
        if (role && content) {
            State.chatHistory.push({ role, content, metadata: null });
        }
    });
    saveActiveConversation();
}

// ============================================
// CHAT SEARCH
// ============================================
function toggleChatSearch() {
    DOM.chatSearchOverlay.classList.toggle('active');
    if (DOM.chatSearchOverlay.classList.contains('active')) {
        DOM.chatSearchField.focus();
        DOM.chatSearchField.value = '';
        searchInChat('');
    } else {
        clearChatSearchHighlights();
    }
}

function searchInChat(query) {
    clearChatSearchHighlights();
    if (!query.trim()) { DOM.chatSearchCount.textContent = '0/0'; return; }

    const messages = DOM.chatScroller.querySelectorAll('.msg-text-content');
    State.chatSearchMatches = [];
    const lower = query.toLowerCase();

    messages.forEach(msg => {
        const text = msg.innerText.toLowerCase();
        if (text.includes(lower)) {
            highlightText(msg, query);
            State.chatSearchMatches.push(msg.closest('.message-row'));
        }
    });

    State.chatSearchIndex = State.chatSearchMatches.length > 0 ? 0 : -1;
    updateSearchCounter();
    if (State.chatSearchMatches.length > 0) scrollToMatch(0);
}

function highlightText(element, query) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    textNodes.forEach(node => {
        const idx = node.textContent.toLowerCase().indexOf(query.toLowerCase());
        if (idx >= 0) {
            const span = document.createElement('span');
            span.className = 'chat-search-highlight';
            const before = node.textContent.substring(0, idx);
            const match = node.textContent.substring(idx, idx + query.length);
            const after = node.textContent.substring(idx + query.length);

            const beforeNode = document.createTextNode(before);
            const matchNode = document.createElement('span');
            matchNode.className = 'chat-search-highlight';
            matchNode.textContent = match;
            const afterNode = document.createTextNode(after);

            const parent = node.parentNode;
            parent.insertBefore(beforeNode, node);
            parent.insertBefore(matchNode, node);
            parent.insertBefore(afterNode, node);
            parent.removeChild(node);
        }
    });
}

function clearChatSearchHighlights() {
    DOM.chatScroller.querySelectorAll('.chat-search-highlight').forEach(el => {
        const parent = el.parentNode;
        parent.insertBefore(document.createTextNode(el.textContent), el);
        parent.removeChild(el);
        parent.normalize();
    });
}

function updateSearchCounter() {
    const total = State.chatSearchMatches.length;
    DOM.chatSearchCount.textContent = total > 0 ? `${State.chatSearchIndex + 1}/${total}` : '0/0';
}

function chatSearchNext() {
    if (State.chatSearchMatches.length === 0) return;
    State.chatSearchIndex = (State.chatSearchIndex + 1) % State.chatSearchMatches.length;
    updateSearchCounter();
    scrollToMatch(State.chatSearchIndex);
}

function chatSearchPrev() {
    if (State.chatSearchMatches.length === 0) return;
    State.chatSearchIndex = (State.chatSearchIndex - 1 + State.chatSearchMatches.length) % State.chatSearchMatches.length;
    updateSearchCounter();
    scrollToMatch(State.chatSearchIndex);
}

function scrollToMatch(index) {
    const el = State.chatSearchMatches[index];
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.style.transition = 'background 0.3s ease';
        el.style.background = 'rgba(14, 165, 233, 0.08)';
        setTimeout(() => { el.style.background = ''; }, 1000);
    }
}

// ============================================
// DRAG & DROP UPLOAD
// ============================================
function initDragDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        DOM.dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        DOM.dropZone.addEventListener(eventName, () => DOM.dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        DOM.dropZone.addEventListener(eventName, () => DOM.dropZone.classList.remove('drag-over'), false);
    });

    DOM.dropZone.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
}

function handleFiles(files) {
    State.uploadQueue = [...State.uploadQueue, ...Array.from(files)];
    renderUploadQueue();
}

function renderUploadQueue() {
    if (State.uploadQueue.length === 0) {
        DOM.uploadFileList.classList.add('hidden');
        DOM.uploadDocBtn.disabled = true;
        return;
    }

    DOM.uploadFileList.classList.remove('hidden');
    DOM.uploadFileList.innerHTML = State.uploadQueue.map((file, i) => `
        <div class="flex items-center justify-between p-3 rounded-xl" style="background: var(--bg-tertiary); border: 1px solid var(--border-color);">
            <div class="flex items-center gap-3 min-w-0">
                <i class="fa-solid fa-file-lines" style="color: var(--accent-sky);"></i>
                <div class="flex flex-col min-w-0">
                    <span class="text-sm truncate" style="color: var(--text-primary);">${escapeHtml(file.name)}</span>
                    <span class="text-[10px]" style="color: var(--text-tertiary);">${formatFileSize(file.size)}</span>
                </div>
            </div>
            <button onclick="removeFromQueue(${i})" class="btn-ghost p-1.5 rounded-lg shrink-0" style="color: var(--accent-rose);">
                <i class="fa-solid fa-xmark text-xs"></i>
            </button>
        </div>
    `).join('');

    DOM.uploadDocBtn.disabled = false;
}

function removeFromQueue(index) {
    State.uploadQueue.splice(index, 1);
    renderUploadQueue();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function handleFileUploadChange(e) {
    if (e.target.files.length) handleFiles(e.target.files);
}

// ============================================
// TAB ROUTING
// ============================================
function switchTab(viewId, activeBtn) {
    DOM.tabBtns.forEach(btn => {
        btn.classList.remove('active');
        btn.style.cssText = '';
    });

    activeBtn.classList.add('active');
    activeBtn.style.cssText = `background: var(--bg-tertiary); color: var(--text-primary); box-shadow: inset 0 0 0 1px var(--border-hover);`;

    DOM.viewPanels.forEach(panel => panel.classList.add('hidden'));
    document.getElementById(viewId).classList.remove('hidden');

    switch (viewId) {
        case 'chat-view':
            DOM.workspaceTitle.textContent = State.chatHistory.length > 0 
                ? (State.conversations[State.currentSessionId]?.title || "Chat Assistant") 
                : "New Conversation";
            DOM.workspaceTag.classList.toggle('hidden', State.chatHistory.length === 0);
            DOM.btnExportChat.classList.toggle('hidden', State.chatHistory.length === 0);
            break;
        case 'indexer-view':
            DOM.workspaceTitle.textContent = "Document Index Control";
            DOM.workspaceTag.classList.add('hidden');
            DOM.btnExportChat.classList.add('hidden');
            break;
        case 'explorer-view':
            DOM.workspaceTitle.textContent = "Vector Retrieval Explorer";
            DOM.workspaceTag.classList.add('hidden');
            DOM.btnExportChat.classList.add('hidden');
            break;
        case 'settings-view':
            DOM.workspaceTitle.textContent = "API Credentials & Settings";
            DOM.workspaceTag.classList.add('hidden');
            DOM.btnExportChat.classList.add('hidden');
            break;
    }

    // Close mobile sidebar
    DOM.sidebar.classList.remove('mobile-open');
    DOM.mobileOverlay.classList.remove('active');
}

// ============================================
// SERVER STATUS & POLLING
// ============================================
async function checkServerStatus(syncConfig = false) {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error('API status call failed');
        const data = await response.json();

        State.indexed = data.indexed;
        updateStatusBadge(data.indexed, data.active_indexing?.status);
        updateIndexerDashboard(data);
        handleIndexingBanner(data.active_indexing);

        if (syncConfig && State.apiKey) syncSettingsWithServer();
    } catch (error) {
        console.error('Server status sync error:', error);
        updateStatusBadge(false, 'offline');
    }
}

function updateStatusBadge(isIndexed, currentIndexingStatus) {
    const activeStages = ['extracting', 'chunking', 'vectorizing', 'indexing'];

    if (activeStages.includes(currentIndexingStatus)) {
        DOM.globalStatusBadge.style.cssText = `background: var(--accent-sky); color: white;`;
        DOM.globalStatusText.innerHTML = `<i class="fa-solid fa-sync fa-spin mr-1"></i> Indexing...`;
    } else if (isIndexed) {
        DOM.globalStatusBadge.style.cssText = `background: var(--accent-emerald); color: white;`;
        DOM.globalStatusText.textContent = "Index Ready";
    } else {
        DOM.globalStatusBadge.style.cssText = `background: var(--accent-amber); color: white;`;
        DOM.globalStatusText.textContent = "Index Offline";
    }
}

function updateIndexerDashboard(data) {
    if (data.pdf_loaded) {
        DOM.idxStatPdf.innerHTML = `<i class="fa-solid fa-circle-check mr-1.5"></i>AI.pdf Available`;
        DOM.idxStatPdf.style.color = 'var(--accent-emerald)';
    } else {
        DOM.idxStatPdf.innerHTML = `<i class="fa-solid fa-circle-xmark mr-1.5"></i>AI.pdf Missing`;
        DOM.idxStatPdf.style.color = 'var(--accent-rose)';
    }

    const activeStages = ['extracting', 'chunking', 'vectorizing', 'indexing'];
    if (data.indexed) {
        DOM.idxStatIndexed.innerHTML = `<i class="fa-solid fa-circle-check mr-1.5"></i>Index Active`;
        DOM.idxStatIndexed.style.color = 'var(--accent-emerald)';
    } else if (activeStages.includes(data.active_indexing?.status)) {
        DOM.idxStatIndexed.innerHTML = `<i class="fa-solid fa-sync fa-spin mr-1.5"></i>Training...`;
        DOM.idxStatIndexed.style.color = 'var(--accent-sky)';
    } else {
        DOM.idxStatIndexed.textContent = "Not Indexed";
        DOM.idxStatIndexed.style.color = 'var(--text-tertiary)';
    }

    DOM.idxStatPages.textContent = data.total_pages || 0;
    DOM.idxStatChunks.textContent = data.total_chunks || 0;
    DOM.idxStatVocab.textContent = `${data.vocabulary_size || 0} words`;
    const faissMb = ((data.faiss_size_bytes || 0) / (1024 * 1024)).toFixed(2);
    DOM.idxStatFaiss.textContent = `${faissMb} MB`;

    DOM.startIndexingBtn.disabled = !data.pdf_loaded || activeStages.includes(data.active_indexing?.status);
}

function handleIndexingBanner(activeIndexing) {
    const activeStages = ['extracting', 'chunking', 'vectorizing', 'indexing'];
    if (activeStages.includes(activeIndexing?.status)) {
        DOM.globalIndexingPanel.classList.remove('hidden');
        State.isIndexingActive = true;

        let labelText = "Processing textbook...";
        if (activeIndexing.status === 'extracting') {
            labelText = `Extracting: Page ${activeIndexing.pages_processed || 0}/${activeIndexing.total_pages || 0}`;
        } else if (activeIndexing.status === 'chunking') {
            labelText = "Chunking text with sliding windows...";
        } else if (activeIndexing.status === 'vectorizing') {
            labelText = `Embedding: Vector ${activeIndexing.current_chunk || 0}/${activeIndexing.total_chunks || 0}`;
        } else if (activeIndexing.status === 'indexing') {
            labelText = "Building FAISS vector index...";
        }

        DOM.alertTitleText.textContent = labelText;
        DOM.alertProgressFill.style.width = `${activeIndexing.progress || 0}%`;
        DOM.alertProgressPercent.textContent = `${activeIndexing.progress || 0}%`;
    } else {
        if (State.isIndexingActive && !DOM.globalIndexingPanel.classList.contains('hidden')) {
            // Indexing just finished
            if (State.confettiEnabled) triggerConfetti();
            showToast('Indexing complete! Vector database is ready.', 'success');
        }
        DOM.globalIndexingPanel.classList.add('hidden');
        State.isIndexingActive = false;
        checkServerStatus(false);
    }
}

function cancelIndexing() {
    if (!confirm('Cancel current indexing operation?')) return;
    showToast('Sending cancel request...', 'info');
    fetch('/api/index/cancel', { method: 'POST' }).catch(() => {});
}

async function syncSettingsWithServer() {
    if (!State.apiKey) return;
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                apiKey: State.apiKey,
                embeddingModel: State.embeddingModel,
                generationModel: State.generationModel,
                temperature: State.temperature,
                topK: State.topK,
                hybridWeight: State.hybridWeight,
                systemPrompt: State.systemPrompt
            })
        });
    } catch (e) { console.error('Settings sync error:', e); }
}

function startStatusPolling() {
    setInterval(() => checkServerStatus(false), 2000);
}

// ============================================
// FORM HANDLERS
// ============================================
async function handleSettingsSubmit(e) {
    e.preventDefault();

    State.apiKey = DOM.settingsApiKey.value.trim();
    State.embeddingModel = DOM.settingsEmbedModel.value;
    State.generationModel = DOM.settingsGenModel.value;
    State.topK = parseInt(DOM.settingsTopK.value);
    State.temperature = parseFloat(DOM.settingsTemperature.value);
    State.hybridWeight = parseFloat(DOM.settingsHybridWeight.value);
    State.systemPrompt = DOM.settingsSystemPrompt.value;
    State.streamingEnabled = document.getElementById('toggle-streaming').classList.contains('active');
    State.autoCitations = document.getElementById('toggle-citations').classList.contains('active');
    State.soundEnabled = document.getElementById('toggle-sound').classList.contains('active');
    State.confettiEnabled = document.getElementById('toggle-confetti').classList.contains('active');

    saveLocalSettings();

    DOM.saveSettingsBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i>Saving...`;

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                apiKey: State.apiKey,
                embeddingModel: State.embeddingModel,
                generationModel: State.generationModel,
                temperature: State.temperature,
                topK: State.topK,
                hybridWeight: State.hybridWeight,
                systemPrompt: State.systemPrompt
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Settings sync failed');
        }

        DOM.saveSettingsBtn.innerHTML = `<i class="fa-solid fa-circle-check text-base"></i><span>Saved!</span>`;
        DOM.saveSettingsBtn.style.background = 'var(--accent-emerald)';
        DOM.saveSettingsBtn.style.color = 'white';
        showToast('Configuration saved and tested', 'success');

        setTimeout(() => {
            DOM.saveSettingsBtn.style.background = '';
            DOM.saveSettingsBtn.style.color = '';
            DOM.saveSettingsBtn.innerHTML = `<i class="fa-solid fa-save text-base"></i><span>Save and Test Configuration</span>`;
        }, 2000);

        checkServerStatus(false);
    } catch (err) {
        console.error(err);
        showToast(`Failed to save: ${err.message}`, 'error');
        DOM.saveSettingsBtn.innerHTML = `<i class="fa-solid fa-save text-base"></i><span>Save and Test Configuration</span>`;
    }
}

async function handleIndexingSubmit(e) {
    e.preventDefault();

    if (!State.apiKey) {
        showToast('API key required. Go to Settings first.', 'warning');
        switchTab('settings-view', document.getElementById('tab-btn-settings'));
        return;
    }

    const size = parseInt(DOM.configChunkSize.value);
    const overlap = parseInt(DOM.configChunkOverlap.value);

    if (overlap >= size) {
        showToast('Overlap must be less than chunk size', 'warning');
        return;
    }

    DOM.startIndexingBtn.disabled = true;
    DOM.startIndexingBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i>Initializing...`;

    try {
        const response = await fetch('/api/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chunkSize: size, chunkOverlap: overlap })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start indexing');
        }

        const data = await response.json();
        if (data.status === 'started' || data.status === 'in_progress') {
            checkServerStatus(false);
            showToast('Indexing started', 'info');
        }
    } catch (err) {
        console.error(err);
        showToast(`Indexing failed: ${err.message}`, 'error');
        DOM.startIndexingBtn.disabled = false;
        DOM.startIndexingBtn.innerHTML = `<i class="fa-solid fa-play-circle text-base"></i><span>Start Vector Training</span>`;
    }
}

async function handleDocUploadSubmit() {
    if (State.uploadQueue.length === 0) return;

    if (!State.apiKey) {
        showToast('API key required for embeddings', 'warning');
        switchTab('settings-view', document.getElementById('tab-btn-settings'));
        return;
    }

    DOM.uploadDocBtn.disabled = true;
    DOM.uploadDocBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i>Uploading...`;

    let totalAdded = 0;

    for (const file of State.uploadQueue) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', { method: 'POST', body: formData });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Upload failed');
            }
            const data = await response.json();
            totalAdded += data.chunks_added || 0;
        } catch (err) {
            showToast(`Failed to upload ${file.name}: ${err.message}`, 'error');
        }
    }

    State.uploadQueue = [];
    renderUploadQueue();

    DOM.uploadStatusMsg.className = 'mt-4 text-xs font-medium py-3 px-4 rounded-xl border flex items-center gap-2';
    DOM.uploadStatusMsg.style.cssText = 'background: rgba(16,185,129,0.08); color: var(--accent-emerald); border-color: rgba(16,185,129,0.2);';
    DOM.uploadStatusMsg.innerHTML = `<i class="fa-solid fa-circle-check text-sm"></i><span>Indexed <strong>${totalAdded}</strong> chunks total.</span>`;
    DOM.uploadStatusMsg.classList.remove('hidden');

    DOM.uploadDocBtn.innerHTML = `<i class="fa-solid fa-upload"></i><span>Upload and Index</span>`;
    checkServerStatus(false);
    showToast(`Uploaded ${totalAdded} chunks`, 'success');
}

async function handleExplorerSearchSubmit(e) {
    e.preventDefault();
    const query = DOM.explorerSearchInput.value.trim();
    if (!query) return;

    if (!State.indexed) {
        showToast('Please build the index first in Document Indexer', 'warning');
        return;
    }

    DOM.explorerResultsGrid.innerHTML = `
        <div class="text-center py-14">
            <i class="fa-solid fa-spinner fa-spin text-4xl mb-4" style="color: var(--accent-sky);"></i>
            <p class="text-sm" style="color: var(--text-tertiary);">Searching vector index...</p>
        </div>
    `;
    DOM.resultsCountBanner.classList.add('hidden');

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, topK: State.topK, hybridWeight: State.hybridWeight })
        });

        if (!response.ok) throw new Error('Search request failed');
        const data = await response.json();
        renderExplorerResults(data.results);
    } catch (err) {
        console.error(err);
        DOM.explorerResultsGrid.innerHTML = `
            <div class="text-center py-14" style="color: var(--accent-rose);">
                <i class="fa-solid fa-circle-exclamation text-4xl mb-4"></i>
                <p class="text-sm">Search failed. Check your API key.</p>
            </div>
        `;
    }
}

function renderExplorerResults(results) {
    DOM.explorerResultsGrid.innerHTML = '';

    if (!results || results.length === 0) {
        DOM.explorerResultsGrid.innerHTML = `
            <div class="text-center py-14" style="color: var(--text-tertiary);">
                <i class="fa-solid fa-magnifying-glass-minus text-4xl mb-4 opacity-30"></i>
                <p class="text-sm">No matching passages found.</p>
            </div>
        `;
        return;
    }

    // Apply filter
    if (State.explorerFilter === 'high-relevance') {
        results = results.filter(r => (r.similarity_score || 0) > 0.8);
    }

    DOM.resultsCountText.textContent = results.length;
    DOM.resultsCountBanner.classList.remove('hidden');

    results.forEach(item => {
        const card = document.createElement('article');
        card.className = 'result-card card rounded-2xl p-5 shadow-xl flex flex-col gap-3 animate-slideIn';
        const pct = Math.round((item.similarity_score || 0) * 100);

        card.innerHTML = `
            <div class="flex justify-between items-center mb-1 select-none">
                <span class="result-page-tag py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center gap-1.5 cursor-pointer" 
                    style="background: var(--bg-tertiary); color: var(--text-tertiary); border: 1px solid var(--border-color);"
                    onmouseenter="showPagePreview(this, ${item.page_start}, '${escapeHtml(item.text?.substring(0, 200) || '')}')"
                    onmouseleave="hidePagePreview()">
                    <i class="fa-solid fa-file-invoice"></i>Page ${item.page_start}
                </span>
                <div class="flex items-center gap-2.5">
                    <span class="text-xs font-bold" style="color: var(--accent-emerald);">${pct}% match</span>
                    <div class="w-14 h-1.5 rounded-full overflow-hidden" style="background: var(--bg-tertiary);">
                        <div class="h-full rounded-full" style="width: ${pct}%; background: var(--accent-emerald);"></div>
                    </div>
                </div>
            </div>
            <div class="text-sm leading-relaxed select-text" style="color: var(--text-secondary);">
                "${highlightQuery(item.text || '', DOM.explorerSearchInput.value)}"
            </div>
            <div class="flex items-center gap-4 text-[10px] font-medium border-t pt-3.5 mt-1 select-none" style="border-color: var(--border-color); color: var(--text-tertiary);">
                <span>Dense: <strong class="font-mono" style="color: var(--text-secondary);">${(item.dense_score || 0).toFixed(3)}</strong></span>
                <span>Sparse: <strong class="font-mono" style="color: var(--text-secondary);">${(item.sparse_score || 0).toFixed(3)}</strong></span>
                <span>Chars: <strong class="font-mono" style="color: var(--text-secondary);">${item.char_count || 0}</strong></span>
            </div>
        `;
        DOM.explorerResultsGrid.appendChild(card);
    });
}

function showPagePreview(el, page, text) {
    const rect = el.getBoundingClientRect();
    DOM.pagePreviewPopup.style.left = `${rect.right + 10}px`;
    DOM.pagePreviewPopup.style.top = `${rect.top}px`;
    DOM.pagePreviewContent.innerHTML = `<strong style="color: var(--accent-sky);">Page ${page}</strong><br><br>${escapeHtml(text)}...`;
    DOM.pagePreviewPopup.classList.add('active');
}

function hidePagePreview() {
    DOM.pagePreviewPopup.classList.remove('active');
}

function highlightQuery(text, query) {
    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 3);
    if (terms.length === 0) return escapeHtml(text);
    let highlighted = escapeHtml(text);
    terms.forEach(term => {
        try {
            const regex = new RegExp(`\b(${term})\b`, 'gi');
            highlighted = highlighted.replace(regex, '<mark>$1</mark>');
        } catch(e) {}
    });
    return highlighted;
}

// ============================================
// CHAT LOGIC
// ============================================
function insertQuickAction(action) {
    const templates = {
        'Summarize': 'Summarize the key points about ',
        'Explain': 'Explain in simple terms: ',
        'Compare': 'Compare and contrast: ',
        'Code': 'Write pseudocode or Python code for: ',
        'Quiz': 'Create a quiz question with answer about: '
    };
    DOM.chatInput.value = templates[action] || '';
    DOM.chatInput.focus();
    autoResizeTextarea();
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const query = DOM.chatInput.value.trim();
    if (!query) return;

    if (!State.indexed) {
        showToast('Textbook not indexed. Go to Document Indexer first.', 'warning');
        return;
    }
    if (!State.apiKey) {
        showToast('API key missing. Go to Settings.', 'warning');
        switchTab('settings-view', document.getElementById('tab-btn-settings'));
        return;
    }

    DOM.chatInput.value = '';
    DOM.chatInput.style.height = '40px';
    DOM.chatSendBtn.disabled = true;

    if (State.soundEnabled) playSound('send');

    const welcome = DOM.chatScroller.querySelector('.assistant-welcome');
    if (welcome) welcome.remove();

    appendMessage('user', query, null, true);

    const typingId = appendTypingIndicator();
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                history: State.chatHistory,
                generationModel: State.generationModel,
                temperature: State.temperature,
                topK: State.topK,
                hybridWeight: State.hybridWeight,
                systemPrompt: State.systemPrompt,
                stream: State.streamingEnabled
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Chat request failed');
        }

        const data = await response.json();
        removeTypingIndicator(typingId);

        if (State.streamingEnabled && data.stream) {
            // Simulate streaming with the full response
            await streamResponse(data);
        } else {
            appendMessage('assistant', data.answer, {
                model: data.model,
                prompt_tokens: data.prompt_tokens,
                completion_tokens: data.completion_tokens,
                retrieved_chunks: data.retrieved_chunks
            }, true);
        }

        if (State.soundEnabled) playSound('receive');

    } catch (err) {
        console.error(err);
        removeTypingIndicator(typingId);
        appendMessage('assistant', `**Error**: ${err.message}\n\nPlease verify your API credentials in Settings.`, null, true);
    } finally {
        DOM.chatSendBtn.disabled = false;
        scrollToBottom();
    }
}

async function streamResponse(data) {
    const metadata = {
        model: data.model,
        prompt_tokens: data.prompt_tokens,
        completion_tokens: data.completion_tokens,
        retrieved_chunks: data.retrieved_chunks
    };

    const bubble = appendMessage('assistant', '', metadata, false);
    const contentDiv = bubble.querySelector('.msg-text-content');
    contentDiv.innerHTML = '<span class="streaming-cursor"></span>';

    const text = data.answer;
    const chars = text.split('');
    let current = '';

    for (let i = 0; i < chars.length; i++) {
        current += chars[i];
        // Batch updates for performance
        if (i % 3 === 0 || i === chars.length - 1) {
            contentDiv.innerHTML = parseTextMarkup(current) + '<span class="streaming-cursor"></span>';
            if (window.hljs) {
                contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
            }
            scrollToBottom();
        }
        // Variable delay for natural feel
        const delay = Math.random() * 8 + 2;
        await new Promise(r => setTimeout(r, delay));
    }

    // Final render
    contentDiv.innerHTML = parseTextMarkup(text);
    if (window.hljs) {
        contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }

    // Add code copy buttons
    contentDiv.querySelectorAll('pre').forEach(pre => {
        if (!pre.querySelector('.code-header')) {
            const header = document.createElement('div');
            header.className = 'code-header';
            const lang = pre.querySelector('code')?.className?.replace('language-', '') || 'code';
            header.innerHTML = `<span>${lang}</span><button class="code-copy-btn" onclick="copyCodeBlock(this)">Copy</button>`;
            pre.insertBefore(header, pre.firstChild);
        }
    });

    // Save to history
    State.chatHistory.push({ role: 'assistant', content: text, metadata });
    saveActiveConversation();
}

function copyCodeBlock(btn) {
    const pre = btn.closest('pre');
    const code = pre.querySelector('code');
    if (code) {
        navigator.clipboard.writeText(code.textContent);
        btn.textContent = 'Copied!';
        btn.style.color = 'var(--accent-emerald)';
        setTimeout(() => { btn.textContent = 'Copy'; btn.style.color = ''; }, 2000);
    }
}

// ============================================
// MESSAGE RENDERING
// ============================================
function appendMessage(role, text, metadata = null, shouldSave = false) {
    const isUser = role === 'user';
    const bubble = document.createElement('div');
    bubble.className = `message-row flex w-full select-text animate-slideIn ${isUser ? 'justify-end' : 'justify-start'}`;
    bubble.dataset.role = role;
    bubble.dataset.timestamp = Date.now();

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let avatarHtml = '';
    if (isUser) {
        avatarHtml = `<div class="size-8 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-xs" style="background: linear-gradient(135deg, var(--accent-purple), var(--accent-sky)); color: white;">JD</div>`;
    } else {
        avatarHtml = `<div class="size-8 rounded-full flex-shrink-0 flex items-center justify-center" style="background: var(--text-primary); color: var(--bg-primary);"><i class="fa-solid fa-sparkles text-[10px]"></i></div>`;
    }

    const parsedText = parseTextMarkup(text);

    let bubbleContent = '';
    if (isUser) {
        bubbleContent = `
            <div class="max-w-[80%] flex items-end gap-3 select-text">
                <div class="msg-text-content bubble-user text-sm leading-5 px-4 py-3 select-text">${parsedText}</div>
                ${avatarHtml}
            </div>
            <div class="msg-time text-right pr-11 mt-1">${timeStr}</div>
        `;
    } else {
        let citationsBlock = '';
        if (State.autoCitations && metadata?.retrieved_chunks?.length > 0) {
            const uniqueId = `cite-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
            const containerId = `cite-box-${Date.now()}`;

            const citationsListHtml = metadata.retrieved_chunks.map(chunk => `
                <div class="p-3 rounded-xl text-xs flex flex-col gap-1.5" style="background: var(--bg-tertiary); border: 1px solid var(--border-color);">
                    <div class="flex justify-between items-center text-[10px] font-bold uppercase tracking-wide" style="color: var(--accent-sky);">
                        <span>Source: Page ${chunk.page_start}</span>
                        <span>Match: ${Math.round((chunk.similarity_score || 0) * 100)}%</span>
                    </div>
                    <div class="italic leading-relaxed" style="color: var(--text-tertiary);">"${escapeHtml((chunk.text || '').substring(0, 300))}..."</div>
                </div>
            `).join('');

            citationsBlock = `
                <div class="mt-3 border-t pt-2.5 w-full select-none" style="border-color: var(--border-color);">
                    <button type="button" class="flex items-center gap-2 text-xs font-semibold transition cursor-pointer" style="color: var(--text-tertiary);" id="${uniqueId}">
                        <i class="fa-solid fa-book-open text-[11px]"></i> 
                        <span>Sources (${metadata.retrieved_chunks.length})</span>
                        <i class="fa-solid fa-chevron-down text-[10px] transition-transform ml-auto"></i>
                    </button>
                    <div class="flex flex-col gap-2 mt-3 hidden select-text" id="${containerId}">
                        ${citationsListHtml}
                    </div>
                </div>
                <div class="flex items-center gap-3.5 mt-2.5 text-[10px] font-semibold select-none" style="color: var(--text-tertiary);">
                    <span>Model: <strong style="color: var(--text-secondary);">${(metadata.model || '').split('/').pop()}</strong></span>
                    <span>Prompt: <strong class="font-mono" style="color: var(--text-secondary);">${metadata.prompt_tokens || 0}</strong></span>
                    <span>Gen: <strong class="font-mono" style="color: var(--text-secondary);">${metadata.completion_tokens || 0}</strong></span>
                </div>
            `;
        }

        bubbleContent = `
            <div class="max-w-[85%] flex flex-col gap-1 select-text">
                <div class="flex items-start gap-3 select-text">
                    ${avatarHtml}
                    <div class="bubble-assistant text-sm leading-5 border border-solid px-4 py-3 select-text">
                        <div class="msg-text-content select-text">${parsedText}</div>
                        ${citationsBlock}
                    </div>
                </div>
                <div class="flex pl-11 items-center gap-1 select-none mt-1 reaction-btn">
                    <button class="btn-ghost size-7 rounded flex items-center justify-center copy-btn" title="Copy"><i class="fa-regular fa-copy text-[11px]"></i></button>
                    <button class="btn-ghost size-7 rounded flex items-center justify-center" title="Thumbs Up" onclick="submitFeedback(this, 1)"><i class="fa-regular fa-thumbs-up text-[11px]"></i></button>
                    <button class="btn-ghost size-7 rounded flex items-center justify-center" title="Thumbs Down" onclick="submitFeedback(this, -1)"><i class="fa-regular fa-thumbs-down text-[11px]"></i></button>
                    <button class="btn-ghost size-7 rounded flex items-center justify-center" title="Regenerate" onclick="regenerateLast()"><i class="fa-solid fa-arrows-rotate text-[11px]"></i></button>
                    <span class="msg-time ml-auto">${timeStr}</span>
                </div>
            </div>
        `;
    }

    bubble.innerHTML = bubbleContent;
    DOM.chatScroller.appendChild(bubble);

    // Bind citation toggle
    if (!isUser && metadata?.retrieved_chunks?.length > 0) {
        const toggleBtn = bubble.querySelector('.mt-3 button');
        const listBlock = bubble.querySelector('.mt-3 > div:last-child');
        const chevron = toggleBtn?.querySelector('.fa-chevron-down');
        if (toggleBtn && listBlock) {
            toggleBtn.addEventListener('click', () => {
                listBlock.classList.toggle('hidden');
                const isOpen = !listBlock.classList.contains('hidden');
                toggleBtn.querySelector('span').textContent = isOpen ? 'Hide Sources' : `Sources (${metadata.retrieved_chunks.length})`;
                if (chevron) chevron.style.transform = isOpen ? 'rotate(180deg)' : '';
            });
        }
    }

    // Bind copy
    if (!isUser) {
        const copyBtn = bubble.querySelector('.copy-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text);
                const icon = copyBtn.querySelector('i');
                icon.className = "fa-solid fa-check text-emerald-400 text-xs";
                setTimeout(() => icon.className = "fa-regular fa-copy text-xs", 2000);
            });
        }
    }

    // Highlight code
    if (!isUser && window.hljs) {
        bubble.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        // Add copy buttons to code blocks
        bubble.querySelectorAll('pre').forEach(pre => {
            if (!pre.querySelector('.code-header')) {
                const header = document.createElement('div');
                header.className = 'code-header';
                const classes = pre.querySelector('code')?.className || '';
                const lang = classes.replace('hljs', '').replace('language-', '').trim() || 'code';
                header.innerHTML = `<span>${lang}</span><button class="code-copy-btn" onclick="copyCodeBlock(this)">Copy</button>`;
                pre.insertBefore(header, pre.firstChild);
            }
        });
    }

    if (shouldSave) {
        State.chatHistory.push({ role, content: text, metadata });
        saveActiveConversation();
    }

    scrollToBottom();
    return bubble;
}

function appendTypingIndicator() {
    const bubbleId = `typing-${Date.now()}`;
    const bubble = document.createElement('div');
    bubble.className = 'flex justify-start w-full animate-slideIn select-none';
    bubble.id = bubbleId;

    bubble.innerHTML = `
        <div class="max-w-[85%] flex flex-col gap-1 select-none">
            <div class="flex items-start gap-3 select-none">
                <div class="size-8 rounded-full flex justify-center items-center flex-shrink-0 select-none" style="background: var(--text-primary); color: var(--bg-primary);">
                    <i class="fa-solid fa-sparkles text-[10px]"></i>
                </div>
                <div class="bubble-assistant text-sm leading-5 border border-solid px-5 py-4 flex items-center shadow-md">
                    <div class="typing-indicator flex items-center gap-1.5">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    DOM.chatScroller.appendChild(bubble);
    return bubbleId;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    DOM.chatScroller.scrollTop = DOM.chatScroller.scrollHeight;
}

function regenerateLast() {
    const lastUser = [...State.chatHistory].reverse().find(m => m.role === 'user');
    if (lastUser) {
        DOM.chatInput.value = lastUser.content;
        DOM.chatForm.requestSubmit();
    }
}

function submitFeedback(btn, val) {
    const row = btn.closest('.message-row');
    const upBtn = row.querySelector('[title="Thumbs Up"]');
    const downBtn = row.querySelector('[title="Thumbs Down"]');

    const upIcon = upBtn.querySelector('i');
    const downIcon = downBtn.querySelector('i');

    if (val === 1) {
        upIcon.className = "fa-solid fa-thumbs-up text-emerald-400 text-[11px]";
        downIcon.className = "fa-regular fa-thumbs-down text-[11px]";
    } else {
        upIcon.className = "fa-regular fa-thumbs-up text-[11px]";
        downIcon.className = "fa-solid fa-thumbs-down text-rose-400 text-[11px]";
    }

    showToast('Feedback recorded', 'success');
}

// ============================================
// MARKDOWN PARSER
// ============================================
function parseTextMarkup(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Code blocks with language
    html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || '';
        return `<pre><code class="language-${language} hljs">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Strikethrough
    html = html.replace(/~~([^~]+)~~/g, '<del>$1</del>');

    // Citations
    html = html.replace(/\[Page\s+(\d+)\]/g, '<span class="book-citation" onclick="showCitationPreview($1)"><i class="fa-solid fa-file-lines"></i> Page $1</span>');
    html = html.replace(/\[Source:\s+Page\s+(\d+)\]/g, '<span class="book-citation" onclick="showCitationPreview($1)"><i class="fa-solid fa-file-lines"></i> Page $1</span>');

    // Blockquotes
    html = html.replace(/^>\s(.+)$/gm, '<blockquote>$1</blockquote>');

    // Headers
    html = html.replace(/^###\s(.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^##\s(.+)$/gm, '<h4>$1</h4>');

    // Lists
    const lines = html.split('\n');
    let inList = false;
    let listType = null;
    let result = [];

    for (const line of lines) {
        const trimmed = line.trim();
        if (/^[-*+]\s/.test(trimmed)) {
            if (!inList || listType !== 'ul') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ul>');
                inList = true;
                listType = 'ul';
            }
            result.push(`<li>${trimmed.substring(2)}</li>`);
        } else if (/^\d+\.\s/.test(trimmed)) {
            if (!inList || listType !== 'ol') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ol>');
                inList = true;
                listType = 'ol';
            }
            result.push(`<li>${trimmed.replace(/^\d+\.\s/, '')}</li>`);
        } else {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = null;
            }
            result.push(line);
        }
    }
    if (inList) result.push(`</${listType}>`);
    html = result.join('\n');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Tables (simple)
    html = html.replace(/\|(.+)\|/g, (match, content) => {
        const cells = content.split('\|').map(c => c.trim()).filter(c => c && !/^[-:]+$/.test(c));
        if (cells.length === 0) return match;
        return '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    });

    // Paragraphs
    html = html.replace(/\n\n/g, '<br><br>');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showCitationPreview(page) {
    showToast(`Loading preview for page ${page}...`, 'info', 2000);
    // In a real app, fetch page content from backend
}

// ============================================
// SOUND EFFECTS
// ============================================
function playSound(type) {
    // Simple oscillator beeps
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (type === 'send') {
            osc.frequency.value = 800;
            gain.gain.value = 0.05;
            osc.start();
            osc.stop(ctx.currentTime + 0.08);
        } else if (type === 'receive') {
            osc.frequency.value = 600;
            gain.gain.value = 0.05;
            osc.start();
            osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.15);
            osc.stop(ctx.currentTime + 0.15);
        } else if (type === 'success') {
            osc.frequency.value = 523;
            gain.gain.value = 0.08;
            osc.start();
            osc.frequency.setValueAtTime(659, ctx.currentTime + 0.1);
            osc.frequency.setValueAtTime(784, ctx.currentTime + 0.2);
            osc.stop(ctx.currentTime + 0.3);
        }
    } catch(e) {}
}

// ============================================
// CONFETTI
// ============================================
function triggerConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const colors = ['#0ea5e9', '#a855f7', '#10b981', '#f59e0b', '#f43f5e', '#38bdf8'];

    for (let i = 0; i < 100; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: -20,
            vx: (Math.random() - 0.5) * 4,
            vy: Math.random() * 3 + 2,
            color: colors[Math.floor(Math.random() * colors.length)],
            size: Math.random() * 6 + 2,
            rotation: Math.random() * 360,
            rotationSpeed: (Math.random() - 0.5) * 10
        });
    }

    let frame = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let active = false;

        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.1;
            p.rotation += p.rotationSpeed;

            if (p.y < canvas.height + 20) active = true;

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation * Math.PI / 180);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.size/2, -p.size/2, p.size, p.size);
            ctx.restore();
        });

        if (active && frame < 200) {
            frame++;
            requestAnimationFrame(animate);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    animate();
    if (State.soundEnabled) playSound('success');
}

// ============================================
// UTILITIES
// ============================================
window.addEventListener('resize', () => {
    const canvas = document.getElementById('confetti-canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
