/**
 * HuoxingTable — 火星员工活跃度名单模块
 *
 * 对外暴露 window.HuoxingTable，提供：
 *   - render(sessionId, huoxingData, messagesContainer, store, formatTime, scrollToBottom)
 *       渲染气泡触发按钮，点击后打开右侧预览面板
 *       huoxingData 格式：
 *         {
 *           "IC": {
 *             "普通员工超低活人员名单": [ {姓名, MIS, 职级}, ... ],
 *             "普通员工待关注名单":     [ {姓名, MIS, 职级}, ... ]
 *           },
 *           "MO": {
 *             "管理者超低活人员名单":   [ {姓名, MIS, 职级}, ... ],
 *             "管理者待关注人员名单":   [ {姓名, MIS, 职级}, ... ]
 *           }
 *         }
 */
(function () {
  'use strict';

  /* ── 内部状态 ────────────────────────────────── */
  const _store = {};          // tableKey → huoxingData
  let _activeTableKey = null; // 当前在面板中展示的 tableKey

  /* ── HTML 转义 ────────────────────────────────── */
  function esc(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* ── 生成单张小表格 HTML ──────────────────────── */
  function generateSmallTableHtml(list, emptyMsg) {
    if (!list || list.length === 0) {
      return `<div style="padding:12px 16px;color:#999;font-size:13px;">${esc(emptyMsg || '暂无数据')}</div>`;
    }
    let rows = list.map((item, idx) => `
      <tr class="table-row" data-index="${idx}">
        <td style="padding:10px 12px;border-bottom:1px solid #f2f3f5;color:#1f2329;font-size:13px;">${esc(item['姓名'] || '-')}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #f2f3f5;color:#4e5969;font-size:12px;">${esc(item['MIS'] || '-')}</td>
        <td style="padding:10px 12px;border-bottom:1px solid #f2f3f5;color:#4e5969;font-size:12px;">${esc(item['职级'] || '-')}</td>
      </tr>
    `).join('');

    return `
      <table style="width:100%;border-collapse:collapse;border:1px solid #e8e8e8;border-radius:6px;overflow:hidden;">
        <thead>
          <tr>
            <th style="background:linear-gradient(180deg,#fafbfc 0%,#f5f6f8 100%);padding:9px 12px;font-size:12px;font-weight:600;color:#4e5969;border-bottom:1px solid #e8e8e8;text-align:left;width:22%;">姓名</th>
            <th style="background:linear-gradient(180deg,#fafbfc 0%,#f5f6f8 100%);padding:9px 12px;font-size:12px;font-weight:600;color:#4e5969;border-bottom:1px solid #e8e8e8;text-align:left;width:28%;">MIS</th>
            <th style="background:linear-gradient(180deg,#fafbfc 0%,#f5f6f8 100%);padding:9px 12px;font-size:12px;font-weight:600;color:#4e5969;border-bottom:1px solid #e8e8e8;text-align:left;width:50%;">部门</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  /* ── 生成分组卡片（IC 或 MO）──────────────────── */
  function generateGroupHtml(groupKey, groupData, panelKey) {
    const isIC = groupKey === 'IC';
    const groupLabel   = isIC ? 'IC（普通员工）' : 'MO（管理者）';
    const groupIcon    = isIC ? '👤' : '👔';
    const highListKey  = isIC ? '普通员工超低活人员名单' : '管理者超低活人员名单';
    const watchListKey = isIC ? '普通员工待关注名单'   : '管理者待关注人员名单';

    const highList  = groupData[highListKey]  || [];
    const watchList = groupData[watchListKey] || [];
    const total = highList.length + watchList.length;

    const toggleId = `hx-toggle-${panelKey}-${groupKey}`;
    const bodyId   = `hx-body-${panelKey}-${groupKey}`;

    return `
      <div class="hx-group-card" id="${toggleId}-card">
        <!-- 折叠头部 -->
        <div class="hx-group-header" onclick="window.HuoxingTable._toggleGroup('${toggleId}')">
          <span class="hx-group-icon">${groupIcon}</span>
          <span class="hx-group-title">${esc(groupLabel)}</span>
          <span class="hx-group-count">${total} 人</span>
          <span class="hx-group-arrow" id="${toggleId}-arrow">▶</span>
        </div>
        <!-- 折叠体 -->
        <div class="hx-group-body" id="${bodyId}" style="display:none;">
          <!-- 超低活表格 -->
          <div class="hx-sub-section">
            <div class="hx-sub-title hx-sub-danger">
              🔴 超低活人员名单
              <span class="hx-sub-count">${highList.length} 人</span>
            </div>
            ${generateSmallTableHtml(highList, '暂无超低活人员')}
          </div>
          <!-- 待关注表格 -->
          <div class="hx-sub-section">
            <div class="hx-sub-title hx-sub-warn">
              🟡 待关注人员名单
              <span class="hx-sub-count">${watchList.length} 人</span>
            </div>
            ${generateSmallTableHtml(watchList, '暂无待关注人员')}
          </div>
        </div>
      </div>
    `;
  }

  /* ── 生成完整面板内容 ─────────────────────────── */
  function generatePanelHtml(huoxingData, panelKey) {
    const icData = huoxingData['IC'] || {};
    const moData = huoxingData['MO'] || {};

    const icTotal = ((icData['普通员工超低活人员名单'] || []).length +
                     (icData['普通员工待关注名单']   || []).length);
    const moTotal = ((moData['管理者超低活人员名单']  || []).length +
                     (moData['管理者待关注人员名单']  || []).length);

    let html = `<div class="hx-panel-content">`;

    if (icTotal + moTotal === 0) {
      html += `<div style="padding:32px;text-align:center;color:#999;">暂无数据</div>`;
    } else {
      if (Object.keys(icData).length > 0) {
        html += generateGroupHtml('IC', icData, panelKey);
      }
      if (Object.keys(moData).length > 0) {
        html += generateGroupHtml('MO', moData, panelKey);
      }
    }

    html += `</div>`;
    return html;
  }

  /* ── 折叠/展开分组 ────────────────────────────── */
  function toggleGroup(toggleId) {
    const bodyId  = toggleId.replace('hx-toggle-', 'hx-body-');
    const arrowId = toggleId + '-arrow';
    const body    = document.getElementById(bodyId);
    const arrow   = document.getElementById(arrowId);
    if (!body) return;

    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'block' : 'none';
    if (arrow) arrow.textContent = isHidden ? '▼' : '▶';
  }

  /* ═══════════════════════════════════════════════
     右侧预览面板（复用 talent_table 的 #preview-table-panel）
     ═══════════════════════════════════════════════ */

  function getPanel()    { return document.getElementById('preview-table-panel'); }
  function getSidebar()  { return document.getElementById('sidebar'); }
  function getChatPanel(){ return document.getElementById('chat-panel'); }

  /* ── 拖拽状态（与 talent_table 独立管理） ────── */
  let _isResizing    = false;
  let _startX        = 0;
  let _startWidth    = 0;
  let _savedPanelWidth = null;

  function _setupResize(handle) {
    handle.addEventListener('mousedown', (e) => {
      const panel     = getPanel();
      const chatPanel = getChatPanel();
      if (!panel || !chatPanel) return;

      _isResizing = true;
      _startX     = e.clientX;
      _startWidth = panel.offsetWidth;

      panel.classList.add('dragging');
      chatPanel.classList.add('dragging');
      document.addEventListener('mousemove', _onResize);
      document.addEventListener('mouseup',   _onResizeStop);
    });
  }

  function _onResize(e) {
    if (!_isResizing) return;
    const panel     = getPanel();
    const chatPanel = getChatPanel();
    if (!panel || !chatPanel) return;
    const diff     = _startX - e.clientX;
    const newWidth = _startWidth + diff;
    const vw       = window.innerWidth;
    const minWidth = vw * 0.3;
    const maxWidth = vw * 0.85;
    if (newWidth >= minWidth && newWidth <= maxWidth) {
      panel.style.width            = newWidth + 'px';
      chatPanel.style.flex         = `0 0 ${vw - newWidth}px`;
      chatPanel.style.width        = `${vw - newWidth}px`;
    }
  }

  function _onResizeStop() {
    _isResizing = false;
    const panel     = getPanel();
    const chatPanel = getChatPanel();
    if (panel)     panel.classList.remove('dragging');
    if (chatPanel) chatPanel.classList.remove('dragging');
    document.removeEventListener('mousemove', _onResize);
    document.removeEventListener('mouseup',   _onResizeStop);
  }

  /**
   * 打开右侧预览面板
   */
  function openPanel(tableKey) {
    const huoxingData = _store[tableKey];
    if (!huoxingData) return;

    _activateButton(tableKey);
    _activeTableKey = tableKey;

    // 隐藏左侧边栏
    const sidebar = getSidebar();
    if (sidebar) sidebar.classList.add('hidden');

    // 聊天区压缩
    const chatPanel = getChatPanel();
    const panel     = getPanel();
    const vw        = window.innerWidth;
    const panelWidthPx = _savedPanelWidth ? parseInt(_savedPanelWidth, 10) : Math.round(vw * 0.55);
    if (chatPanel) {
      chatPanel.classList.add('split');
      chatPanel.style.flex  = `0 0 ${vw - panelWidthPx}px`;
      chatPanel.style.width = `${vw - panelWidthPx}px`;
    }

    if (panel) {
      // 计算总人数
      const icData  = huoxingData['IC'] || {};
      const moData  = huoxingData['MO'] || {};
      const icTotal = ((icData['普通员工超低活人员名单'] || []).length +
                       (icData['普通员工待关注名单']   || []).length);
      const moTotal = ((moData['管理者超低活人员名单']  || []).length +
                       (moData['管理者待关注人员名单']  || []).length);

      // 更新标题
      const titleEl = panel.querySelector('.ptp-title');
      if (titleEl) titleEl.textContent = `活性待关注人员名单（共 ${icTotal + moTotal} 人）`;

      // 隐藏下载按钮（此模块暂不支持下载）
      const downloadBtn = document.getElementById('preview-table-download');
      if (downloadBtn) downloadBtn.style.display = 'none';

      // 渲染内容
      const tableSection = panel.querySelector('.preview-table-content');
      if (tableSection) {
        tableSection.innerHTML = generatePanelHtml(huoxingData, tableKey);
      }

      // 添加拖拽把手（仅首次）
      if (!panel.querySelector('.preview-panel-resize-handle')) {
        const handle = document.createElement('div');
        handle.className = 'preview-panel-resize-handle';
        panel.insertBefore(handle, panel.firstChild);
        _setupResize(handle);
      }

      panel.style.width = panelWidthPx + 'px';
      panel.classList.add('visible');
    }
  }

  /**
   * 关闭面板（由 close 按钮或 ESC 触发）
   */
  function closePanel() {
    const sidebar   = getSidebar();
    const chatPanel = getChatPanel();
    const panel     = getPanel();

    _deactivateButton(_activeTableKey);

    if (sidebar)   sidebar.classList.remove('hidden');
    if (chatPanel) {
      chatPanel.classList.remove('split');
      chatPanel.style.flex  = '';
      chatPanel.style.width = '';
    }
    if (panel) {
      _savedPanelWidth = panel.style.width || null;
      panel.classList.remove('visible');
      panel.style.width = '';
    }

    // 恢复下载按钮
    const downloadBtn = document.getElementById('preview-table-download');
    if (downloadBtn) downloadBtn.style.display = '';

    _activeTableKey = null;
    _isResizing     = false;
  }

  function _activateButton(tableKey) {
    _deactivateButton(_activeTableKey);
    const btn = document.querySelector(`.hx-table-trigger[data-hx-key="${tableKey}"]`);
    if (btn) {
      btn.closest('.msg-bubble')?.classList.add('preview-active');
      const arrowEl = btn.querySelector('.ttt-arrow');
      if (arrowEl) arrowEl.textContent = '● 预览中';
    }
  }

  function _deactivateButton(tableKey) {
    if (!tableKey) return;
    const btn = document.querySelector(`.hx-table-trigger[data-hx-key="${tableKey}"]`);
    if (btn) {
      btn.closest('.msg-bubble')?.classList.remove('preview-active');
      const arrowEl = btn.querySelector('.ttt-arrow');
      if (arrowEl) arrowEl.textContent = '▶ 查看';
    }
  }

  /* ── 计算总人数 ───────────────────────────────── */
  function countTotal(huoxingData) {
    const icData  = (huoxingData || {})['IC'] || {};
    const moData  = (huoxingData || {})['MO'] || {};
    return ((icData['普通员工超低活人员名单'] || []).length +
            (icData['普通员工待关注名单']   || []).length +
            (moData['管理者超低活人员名单']  || []).length +
            (moData['管理者待关注人员名单']  || []).length);
  }

  /**
   * 渲染气泡触发按钮
   */
  function render(sessionId, huoxingData, messagesContainer, chatStore, formatTime, scrollToBottom) {
    const tableKey = `hx_${Date.now()}`;
    _store[tableKey] = huoxingData;

    const total = countTotal(huoxingData);

    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.innerHTML = `
      <div class="msg-avatar bot-avatar">🤖</div>
      <div class="msg-bubble" style="padding:0;overflow:hidden;">
        <button class="hx-table-trigger talent-table-trigger" data-hx-key="${tableKey}"
                onclick="window.HuoxingTable.open('${tableKey}')">
          <span class="ttt-body">
            <div class="ttt-title">活性名单：共 ${total} 人需关注</div>
            <div class="ttt-desc">点击查看 IC / 管理者超低活及待关注名单</div>
          </span>
          <span class="ttt-arrow">▶ 查看</span>
        </button>
      </div>
      <div class="msg-time">${formatTime()}</div>
    `;
    messagesContainer.appendChild(row);

    chatStore.addMessage(sessionId, {
      role: 'bot',
      type: 'huoxing_table',
      content: `活性待关注名单：共 ${total} 人需关注（点击查看）`,
      huoxingData: huoxingData,
      tableKey: tableKey,
    });

    scrollToBottom(true);
  }

  /* ── 公开 API ───────────────────────────────── */
  window.HuoxingTable = {
    render,
    open:           (key) => openPanel(key),
    close:          closePanel,
    _toggleGroup:   toggleGroup,
    _storeData:     (key, data) => { _store[key] = data; },
    _getActiveKey:  () => _activeTableKey,
  };
})();
