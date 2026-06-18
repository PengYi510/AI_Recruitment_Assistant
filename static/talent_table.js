/**
 * TalentTable — 人才表格模块
 *
 * 对外暴露 window.TalentTable，提供：
 *   - render(sessionId, tableData, messagesContainer, store, formatTime, scrollToBottom)
 *       渲染气泡触发按钮，点击后打开右侧预览面板
 *   - open(key)            打开右侧预览面板
 *   - close()              关闭右侧预览面板
 *   - bindCloseEvents()    绑定关闭事件（close 按钮、ESC）
 *   - _toggleReasonDetail(el, event)  折叠/展开推荐理由详情
 */
(function () {
  'use strict';

  /* ── 内部状态 ────────────────────────────────── */
  const _store = {};          // tableKey → talentList
  let _activeTableKey = null; // 当前在面板中展示的 tableKey

  /* ── 拖拽状态 ────────────────────────────────── */
  let _isResizing = false;
  let _startX = 0;
  let _startWidth = 0;
  let _savedPanelWidth = null;
  let _savedMainFlex = null;
  let _permissionData = null;

  /* ── HTML 转义 ────────────────────────────────── */
  function esc(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* ── 推荐理由详情折叠/展开 ──────────────────── */
  function toggleReasonDetail(element, event) {
    event.stopPropagation();
    const arrow   = element.querySelector('.reason-detail-arrow');
    const content = element.nextElementSibling;
    const expanded = content.classList.toggle('expanded');
    arrow.classList.toggle('expanded', expanded);
  }

  /* ── 格式化推荐理由详情（按【SPLIT】分割） ──── */
  function formatReasonDoc(reasonDoc, storyDoc) {
    if (!reasonDoc && !storyDoc) {
        return '';
    }
    let html = '';
    if (reasonDoc) {
      const items = reasonDoc.split('【SPLIT】').map(s => s.trim()).filter(Boolean);
      html = items.map(item => {
        const m = item.match(/\(数据来源:\s*([^)]+)\)/);
        if (m) {
          const body = item.replace(/\(数据来源:\s*[^)]+\)/, '').trim();
          const source = m[1];
          return `<div class="reason-detail-item">
            <span class="reason-detail-source">数据来源: ${esc(source)}</span>
            <div>${esc(body)}</div>
          </div>`;
        }
        return `<div class="reason-detail-item">${esc(item)}</div>`;
      }).join('');
    }
    // 如果有故事线，追加查看按钮
    if (storyDoc && storyDoc.trim() && storyDoc.trim() !== '无') {
      html += `<button class="story-line-btn" onclick="window.TalentTable._showStoryPopup(event, this)" data-story="${esc(storyDoc)}"><span>点击查看故事线</span></button>`;
    }

    return html;
  }

  /* ── 生成表格 HTML ──────────────────────────── */
  function generateTableHtml(talentList, reqType) {
    if (!talentList || talentList.length === 0) {
      return '<div style="padding:32px;text-align:center;color:#999;">暂无数据</div>';
    }
    console.log("talentList is ", talentList);

    // 判断是否所有人都没有命中条件
    const hasAnyHitConditions = talentList.some(talent => {
      const hitConds = talent.hit_conditions || [];
      return hitConds.length > 0;
    });

    // 根据是否有命中条件动态调整表头和 colspan
    const colspanValue = hasAnyHitConditions ? '4' : '3';
    const hitConditionHeader = hasAnyHitConditions
      ? '<th class="col-hit">命中条件</th>'
      : '';

    let html = `
      <table class="talent-table-wrapper">
        <thead>
          <tr>
            <th class="col-name">姓名/MIS</th>
            <th class="col-basic-info">基本信息</th>
            ${hitConditionHeader}
            <th class="col-reason">推荐理由</th>
          </tr>
        </thead>
        <tbody>
    `;

    talentList.forEach((talent, index) => {
      const nameMis   = talent.name_mis            || {};
      const basicInfo = talent.basic_info           || {};
      const hitConds  = talent.hit_conditions       || [];
      const reason    = talent.recommend_reason     || '';
      const reasonDoc = talent.recommend_reason_doc || '';

      const basicInfoHtml = `
        <div>序列：${esc(basicInfo.序列 || '-')}</div>
        <div>职级：${esc(basicInfo.职级 || '-')}</div>
        <div>部门：${esc(basicInfo.部门 || '-')}</div>
        <div>城市：${esc(basicInfo.城市 || '-')}</div>
      `;

      const conditionsHtml = hitConds.length > 0
        ? hitConds.map(c => {
            const vals = (c.values || []).map(v => esc(String(v))).join('、');
            return `<div><strong>${esc(c.condition_name || '')}:</strong> ${vals}</div>`;
          }).join('')
        : '<div>-</div>';

      let storyDoc = talent.story_doc || '';
      if (storyDoc === '无'){storyDoc = '';}

      const reasonDetailHtml = (reasonDoc || storyDoc) ? `
        <div class="reason-detail-container">
          <div class="reason-detail-toggle"
               onclick="window.TalentTable._toggleReasonDetail(this, event)">
            <span class="reason-detail-arrow">▼</span>
            <span>点击查看更多信息</span>
          </div>
          <div class="reason-detail-content">${formatReasonDoc(reasonDoc, storyDoc)}</div>
        </div>
      ` : '';

      // 根据是否有命中条件决定是否显示该列
      const hitConditionCell = hasAnyHitConditions
        ? `<td class="col-hit"><div class="conditions-cell">${conditionsHtml}</div></td>`
        : '';

      const empno = nameMis.工号 || '';
      const empnoHtml = (reqType === 3 && empno)
        ? `<a class="t-empno-link" href="https://zhaopin.sankuai.com/resume-details?resumeId=${esc(empno)}" target="_blank" rel="noopener noreferrer">${esc(empno)}</a>`
        : esc(empno);

      html += `
        <tr class="table-row" data-index="${index}">
          <td class="col-name">
            <div class="talent-name-cell">
              <div class="t-name">${esc(nameMis.姓名 || '未知')}</div>
              <div class="t-mis">${esc(nameMis.mis || '')}</div>
              <div class="t-empno">${empnoHtml}</div>
            </div>
          </td>
          <td class="col-basic-info"><div class="text-cell">${basicInfoHtml}</div></td>
          ${hitConditionCell}
          <td class="col-reason">
            <div class="reason-cell">${esc(reason)}</div>
            ${reasonDetailHtml}
          </td>
        </tr>
      `;
    });

    html += `
          <tr class="no-more-row"><td colspan="${colspanValue}">─── 没有更多了 ───</td></tr>
        </tbody>
      </table>
    `;
    return html;
  }

  /* ═══════════════════════════════════════════════
     右侧预览面板
     ═══════════════════════════════════════════════ */

  /** 获取关键 DOM 引用 */
  function getPanel()    { return document.getElementById('preview-table-panel'); }
  function getSidebar()  { return document.getElementById('sidebar'); }
  function getChatPanel(){ return document.getElementById('chat-panel'); }

  /**
   * 打开右侧预览面板
   * @param {string} tableKey 表格数据 key
   */
  function openPanel(tableKey) {
    const storeEntry = _store[tableKey];
    if (!storeEntry) return;
    const talentList = storeEntry.talentList || storeEntry;  // 兼容旧格式
    const reqType = storeEntry.reqType !== undefined ? storeEntry.reqType : null;

    // 1) 将按钮标记为激活态
    _activateButton(tableKey);
    
    _activeTableKey = tableKey;

    // 2) 隐藏左侧边栏
    const sidebar = getSidebar();
    if (sidebar) sidebar.classList.add('hidden');

    // 3) 聊天区压缩到左半部分（用 JS 直接设置像素值，不用 CSS !important）
    const chatPanel = getChatPanel();
    const panel = getPanel();
    const vw = window.innerWidth;
    // 如果有上次拖拽保存的宽度，优先使用；否则默认 60%
    const panelWidthPx = _savedPanelWidth ? parseInt(_savedPanelWidth, 10) : Math.round(vw * 0.6);
    if (chatPanel) {
      chatPanel.classList.add('split');
      chatPanel.style.flex = `0 0 ${vw - panelWidthPx}px`;
      chatPanel.style.width = `${vw - panelWidthPx}px`;
    }

    // 4) 渲染表格到面板
    if (panel) {
      const tableSection = panel.querySelector('.preview-table-content');
      if (tableSection) {
        tableSection.innerHTML = generateTableHtml(talentList, reqType);
      }

      // 设置面板标题
      const titleEl = panel.querySelector('.ptp-title');
      if (titleEl) titleEl.textContent = `候选人列表（共 ${talentList.length} 位）`;

      // 添加拖拽把手（仅首次）
      if (!panel.querySelector('.preview-panel-resize-handle')) {
        const handle = document.createElement('div');
        handle.className = 'preview-panel-resize-handle';
        panel.insertBefore(handle, panel.firstChild);
        _setupResize(handle);
      }

      // 显示面板（用像素值而非 CSS 百分比）
      panel.style.width = panelWidthPx + 'px';
      panel.classList.add('visible');
    }
  }

  /**
   * 关闭右侧预览面板
   */
  function closePanel() {
    const sidebar = getSidebar();
    const chatPanel = getChatPanel();
    const panel = getPanel();

    // 恢复按钮状态
    _deactivateButton(_activeTableKey);

    // 显示左侧边栏
    if (sidebar) sidebar.classList.remove('hidden');

    // 聊天区恢复全宽
    if (chatPanel) {
      chatPanel.classList.remove('split');
      chatPanel.style.flex = '';
      chatPanel.style.width = '';
    }

    // 隐藏面板
    if (panel) {
      _savedPanelWidth = panel.style.width || null;
      panel.classList.remove('visible');
      panel.style.width = '';
      _savedMainFlex = null;
    }

    _activeTableKey = null;
    _isResizing = false;
  }

  /**
   * 激活对应按钮（高亮 + 文案切换）
   */
  function _activateButton(tableKey) {
    // 先恢复之前激活的按钮
    _deactivateButton(_activeTableKey);

    const btn = document.querySelector(`.talent-table-trigger[data-table-key="${tableKey}"]`);
    if (btn) {
      btn.closest('.msg-bubble')?.classList.add('preview-active');
      const arrowEl = btn.querySelector('.ttt-arrow');
      if (arrowEl) arrowEl.textContent = '● 预览中';
    }
  }

  /**
   * 恢复按钮为默认状态
   */
  function _deactivateButton(tableKey) {
    if (!tableKey) return;
    const btn = document.querySelector(`.talent-table-trigger[data-table-key="${tableKey}"]`);
    if (btn) {
      btn.closest('.msg-bubble')?.classList.remove('preview-active');
      const arrowEl = btn.querySelector('.ttt-arrow');
      if (arrowEl) arrowEl.textContent = '▶ 查看';
    }
  }

  /* ── 拖拽调整面板宽度 ────────────────────────── */
  function _setupResize(handle) {
    handle.addEventListener('mousedown', (e) => {
      const panel = getPanel();
      const chatPanel = getChatPanel();
      if (!panel || !chatPanel) return;

      _isResizing = true;
      _startX = e.clientX;
      _startWidth = panel.offsetWidth;

      panel.classList.add('dragging');
      chatPanel.classList.add('dragging');

      document.addEventListener('mousemove', _onResize);
      document.addEventListener('mouseup', _onResizeStop);
    });
  }

  function _onResize(e) {
    if (!_isResizing) return;
    const panel = getPanel();
    const chatPanel = getChatPanel();
    if (!panel || !chatPanel) return;

    const diff = _startX - e.clientX;
    const newWidth = _startWidth + diff;
    const vw = window.innerWidth;

    // 面板最小 30%，最大 85%
    const minWidth = vw * 0.3;
    const maxWidth = vw * 0.85;

    if (newWidth >= minWidth && newWidth <= maxWidth) {
      panel.style.width = newWidth + 'px';
      // 聊天区 = 视窗宽度 - 面板实际宽度，确保两者始终填满整个屏幕
      chatPanel.style.flex = `0 0 ${vw - newWidth}px`;
      chatPanel.style.width = `${vw - newWidth}px`;
    }
  }

  function _onResizeStop() {
    _isResizing = false;
    const panel = getPanel();
    const chatPanel = getChatPanel();
    if (panel) panel.classList.remove('dragging');
    if (chatPanel) chatPanel.classList.remove('dragging');
    document.removeEventListener('mousemove', _onResize);
    document.removeEventListener('mouseup', _onResizeStop);
  }

  function showToast(msg, type = 'info', durationMs = 3000) {
    const el = document.createElement('div');
    const toastWrap = document.getElementById('toast-wrap');
    el.className = 'toast' + (type === 'error' ? ' toast-error' : '');
    el.textContent = msg;
    toastWrap.appendChild(el);
    setTimeout(() => {
      el.style.animation = 'toastOut .25s ease forwards';
      setTimeout(() => el.remove(), 260);
    }, durationMs);
  }

  /* ── 检测表格字段 ──────────────────────────── */
  function detectTableFields(talentList) {
    // 定义可能的字段及其信息
    const fieldDefinitions = [
      { key: 'name_mis.姓名', label: '姓名', width: 10, getter: t => (t.name_mis || {})['姓名'] || '' },
      { key: 'name_mis.mis', label: 'MIS', width: 16, getter: t => (t.name_mis || {})['mis'] || '' },
      { key: 'name_mis.工号', label: '工号', width: 12, getter: t => (t.name_mis || {})['工号'] || '' },
      { key: 'basic_info.序列', label: '序列', width: 14, getter: t => (t.basic_info || {})['序列'] || '' },
      { key: 'basic_info.职级', label: '职级', width: 8, getter: t => (t.basic_info || {})['职级'] || '' },
      { key: 'basic_info.部门', label: '部门', width: 10, getter: t => (t.basic_info || {})['部门'] || '' },
      { key: 'basic_info.城市', label: '城市', width: 10, getter: t => (t.basic_info || {})['城市'] || '' },
      { key: 'hit_conditions', label: '命中条件', width: 40, getter: t => {
        const conds = t.hit_conditions || [];
        return conds.length > 0 ? conds.map(c => `${c.condition_name}: ${(c.values || []).join('、')}`).join('; ') : '';
      }},
      { key: 'recommend_reason', label: '推荐理由', width: 50, getter: t => t.recommend_reason || '' },
      { key: 'story_doc', label: '故事线', width: 50, getter: t => t.story_doc || '' },
    ];

    // 检测哪些字段实际存在于数据中
    const activeFields = [];
    for (const fieldDef of fieldDefinitions) {
      // 检查至少一条数据中该字段是否有值
      const hasData = talentList.some(item => {
        const value = fieldDef.getter(item);
        return value && value.toString().trim() !== '';
      });
      
      if (hasData) {
        activeFields.push(fieldDef);
      }
    }

    // 如果没有检测到任何字段，返回所有基础字段
    if (activeFields.length === 0) {
      return fieldDefinitions.filter(f => 
        ['name_mis.姓名', 'name_mis.mis', 'name_mis.工号', 'basic_info.序列', 'basic_info.职级'].includes(f.key)
      );
    }

    return activeFields;
  }

  /* ── 下载表格为 Excel (.xlsx) ──────────────── */
  async function downloadTableAsExcel() {
    try {
      // 首先检查用户权限
      if (!_permissionData){
          const permissionRes = await fetch('/api/check-download-permission', {
          method: 'GET',
          credentials: 'include',
        });

        if (!permissionRes.ok) {
          showToast('❌ 请先登录');
          return;
        }
        _permissionData = await permissionRes.json();
      }

      if (!_permissionData.has_download_permission) {
        // 用户无权限
        showToast(`❌ ${_permissionData.download_permission_message}`);
        return;
      }

      // 权限检查通过，继续下载
      if (!_activeTableKey || !_store[_activeTableKey]) return;
      const talentList = _store[_activeTableKey];
      if (!talentList || talentList.length === 0) return;

      // 💡 动态检测表格字段
      const activeFields = detectTableFields(talentList);
      const headers = activeFields.map(f => f.label);
      const rows = [headers];

      // 根据检测到的字段构建数据行
      talentList.forEach(t => {
        const row = activeFields.map(field => field.getter(t));
        rows.push(row);
      });

      // 创建工作簿和工作表
      const wb = XLSX.utils.book_new();
      const ws = XLSX.utils.aoa_to_sheet(rows);

      // 💡 动态设置列宽
      ws['!cols'] = activeFields.map(f => ({ wch: f.width }));

      XLSX.utils.book_append_sheet(wb, ws, '候选人列表');

      // 导出文件
      const now = new Date();
      const ts = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')} ${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
      const fileName = `候选人列表_${ts}.xlsx`;
      XLSX.writeFile(wb, fileName);
      
      showToast('✅ Excel 表格下载成功');
    } catch (error) {
      console.error('下载表格出错:', error);
      showToast('❌ 下载表格失败，请重试');
    }
  }

  /* ── 统一关闭面板（兼容 HuoxingTable） ──────── */
  function smartClose() {
    // 如果 HuoxingTable 正在展示，委托给它关闭
    if (window.HuoxingTable && window.HuoxingTable._getActiveKey()) {
      window.HuoxingTable.close();
    } else {
      closePanel();
    }
  }

  /* ── 绑定关闭事件（App 启动时调用一次） ─────── */
  function bindCloseEvents() {
    // 面板关闭按钮
    const closeBtn = document.getElementById('preview-table-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', smartClose);
    }

    // 面板下载按钮
    const downloadBtn = document.getElementById('preview-table-download');
    if (downloadBtn) {
      downloadBtn.addEventListener('click', downloadTableAsExcel);
    }

    // ESC 关闭
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        // 优先关闭故事线弹窗
        const storyOverlay = document.getElementById('talent-story-popup-overlay');
        if (storyOverlay && storyOverlay.classList.contains('show')) {
          closeStoryPopup();
        } else if (window.HuoxingTable && window.HuoxingTable._getActiveKey()) {
          window.HuoxingTable.close();
        } else if (_activeTableKey) {
          closePanel();
        }
      }
    });
  }

  /* ── 故事线弹窗 ──────────────────────────────── */
  function showStoryPopup(event, button) {
    event.stopPropagation();
    const storyDoc = button.getAttribute('data-story');
    if (!storyDoc) return;

    let overlay = document.getElementById('talent-story-popup-overlay');
    if (!overlay) return;

    document.getElementById('talent-story-popup-text').innerHTML =
      storyDoc.replace(/\n/g, '<br>');
    overlay.classList.add('show');
    document.body.style.overflow = 'hidden';
  }

  function closeStoryPopup() {
    const overlay = document.getElementById('talent-story-popup-overlay');
    if (overlay) overlay.classList.remove('show');
    // 只有面板也关闭时才恢复滚动
    if (!_activeTableKey) {
      document.body.style.overflow = '';
    }
  }

  /**
   * 渲染人才表格气泡（预览按钮）
   * @param {string}   sessionId         当前会话 ID
   * @param {object}   tableData         SSE 返回的 talent_table 数据
   * @param {Element}  messagesContainer 消息列表 DOM 节点
   * @param {object}   chatStore         ChatStore 实例（用于写入历史）
   * @param {Function} formatTime        时间格式化函数
   * @param {Function} scrollToBottom    滚动到底部函数
   */
  function render(sessionId, tableData, messagesContainer, chatStore, formatTime, scrollToBottom) {
    const talentList = tableData.content || [];
    const totalNum   = tableData.num_dict?.total_num || talentList.length;

    const tableKey = `tt_${Date.now()}`;
    _store[tableKey] = { talentList, reqType: tableData.req_type };

    const row = document.createElement('div');
    row.className = 'msg-row bot';
    row.innerHTML = `
      <div class="msg-avatar bot-avatar">🤖</div>
      <div class="msg-bubble" style="padding:0;overflow:hidden;">
        <button class="talent-table-trigger" data-table-key="${tableKey}"
                onclick="window.TalentTable.open('${tableKey}')">
          <span class="ttt-body">
            <div class="ttt-title">找到 ${totalNum} 位候选人</div>
            <div class="ttt-desc">点击查看完整候选人信息表格</div>
          </span>
          <span class="ttt-arrow">▶ 查看</span>
        </button>
      </div>
      <div class="msg-time">${formatTime()}</div>
    `;
    messagesContainer.appendChild(row);

    // 保存完整的表格数据到历史记录
    chatStore.addMessage(sessionId, {
      role: 'bot',
      type: 'talent_table',
      content: `📊 找到 ${totalNum} 位候选人（点击查看表格）`,
      tableData: tableData,
      tableKey: tableKey,
    });

    scrollToBottom(true);
  }

  /* ── 公开 API ───────────────────────────────── */
  window.TalentTable = {
    render,
    bindCloseEvents,
    open:                 (key) => openPanel(key),
    close:                closePanel,
    _toggleReasonDetail:  toggleReasonDetail,
    _showStoryPopup:      showStoryPopup,
    _closeStoryPopup:     closeStoryPopup,
    _storeTableData:      (key, data) => {
      // data 可能是 talentList 数组（旧格式）或含 req_type 的 tableData 对象
      if (Array.isArray(data)) {
        _store[key] = { talentList: data, reqType: null };
      } else {
        _store[key] = { talentList: data.content || [], reqType: data.req_type };
      }
    },  // 供历史恢复使用
  };
})();
