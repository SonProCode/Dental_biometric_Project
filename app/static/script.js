/* ══════════════════════════════════════════════════════════════
   DentalID – Frontend Logic
   ══════════════════════════════════════════════════════════════ */

// ── Utilities ─────────────────────────────────────────────────

function showLoading(text = 'Processing…') {
  document.getElementById('loading-text').textContent = text;
  document.getElementById('loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

function showToast(msg, type = 'success', duration = 3500) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add('hidden'), duration);
}

function showMessage(id, msg, type = 'error') {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = `message ${type}`;
}
function hideMessage(id) {
  document.getElementById(id).classList.add('hidden');
}

function readFileAsDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

// ── Tab navigation ────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`panel-${tab}`).classList.add('active');

    if (tab === 'delete') refreshDeleteDropdown();
  });
});

// ── Drag-over highlight ───────────────────────────────────────

document.querySelectorAll('.upload-zone').forEach(zone => {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', () => zone.classList.remove('drag-over'));
});

// ══════════════════════════════════════════════════════════════
// 1. RECOGNIZE PANEL
// ══════════════════════════════════════════════════════════════

const recFileInput  = document.getElementById('rec-file');
const recPreview    = document.getElementById('rec-preview');
const recPreviewWrap= document.getElementById('rec-preview-wrap');
const recUploadLabel= document.querySelector('#panel-recognize .upload-label');
const recClearBtn   = document.getElementById('rec-clear-btn');
const recBtn        = document.getElementById('rec-btn');
const recResults    = document.getElementById('rec-results');

recFileInput.addEventListener('change', async () => {
  const file = recFileInput.files[0];
  if (!file) return;
  const url = await readFileAsDataURL(file);
  recPreview.src = url;
  recPreviewWrap.classList.remove('hidden');
  recUploadLabel.classList.add('hidden');
  hideMessage('rec-message');
  recResults.classList.add('hidden');
});

recClearBtn.addEventListener('click', e => {
  e.stopPropagation();
  recFileInput.value = '';
  recPreviewWrap.classList.add('hidden');
  recUploadLabel.classList.remove('hidden');
  recResults.classList.add('hidden');
  hideMessage('rec-message');
});

recBtn.addEventListener('click', async () => {
  const file = recFileInput.files[0];
  if (!file) { showToast('Vui lòng chọn ảnh X-ray trước.', 'error'); return; }

  hideMessage('rec-message');
  recResults.classList.add('hidden');
  showLoading('🔍 Đang phân tích X-ray…');

  try {
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch('/recognize', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok || !data.success) {
      showMessage('rec-message', '⚠️ ' + (data.detail || data.message || 'Lỗi không xác định.'), 'info');
      return;
    }

    renderRecognizeResults(data);

  } catch (err) {
    showMessage('rec-message', '❌ Lỗi kết nối server: ' + err.message, 'error');
  } finally {
    hideLoading();
  }
});

function renderRecognizeResults(data) {
  const isUnknown = data.result === 'unknown';
  const badge = document.getElementById('rec-badge');
  const nameEl = document.getElementById('rec-person-name');
  const scoreCircle = document.getElementById('rec-score-circle');

  // Update badge and text for Unknown state
  if (isUnknown) {
    badge.textContent = '?';
    badge.classList.add('unknown');
    nameEl.textContent = 'Unknown (Không khớp)';
    nameEl.classList.add('unknown-text');
  } else {
    badge.textContent = '✓';
    badge.classList.remove('unknown');
    nameEl.textContent = data.result;
    nameEl.classList.remove('unknown-text');
  }

  document.getElementById('rec-tooth-count').textContent = `${data.tooth_count} teeth detected`;

  // Score display
  const pct = Math.round(data.score * 100);
  scoreCircle.textContent = `${pct}%`;
  scoreCircle.style.setProperty('--pct', pct);
  if (isUnknown) {
    scoreCircle.classList.add('low-confidence');
  } else {
    scoreCircle.classList.remove('low-confidence');
  }

  // Top-3 table
  const tbody = document.getElementById('rec-top3-body');
  tbody.innerHTML = '';
  const medals = ['🥇', '🥈', '🥉'];
  (data.top3 || []).forEach((r, i) => {
    const pct = Math.round(r.score * 100);
    const barW = Math.max(4, pct);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="rank-${i+1}">${medals[i] || i+1}</span></td>
      <td>${escHtml(r.name)}</td>
      <td><strong>${pct}%</strong></td>
      <td>
        <span class="score-bar" style="width:${barW}px"></span>
        <small style="color:var(--text-muted)">${r.num_matched} pairs</small>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Tooth crops
  const grid = document.getElementById('tooth-grid');
  grid.innerHTML = '';
  document.getElementById('crop-count').textContent = data.tooth_count;
  (data.tooth_images || []).forEach((b64, i) => {
    const div = document.createElement('div');
    div.className = 'tooth-item';
    div.innerHTML = `
      <img src="data:image/png;base64,${b64}" alt="Tooth ${i+1}" loading="lazy" />
      <div class="tooth-label">Răng ${i+1}</div>
    `;
    div.addEventListener('click', () => openLightbox(`data:image/png;base64,${b64}`, `Răng ${i+1}`));
    grid.appendChild(div);
  });

  recResults.classList.remove('hidden');
}

// ── Lightbox (quick zoom) ─────────────────────────────────────
function openLightbox(src, label) {
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position:fixed;inset:0;z-index:99999;
    background:rgba(0,0,0,0.85);backdrop-filter:blur(8px);
    display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:zoom-out;
  `;
  overlay.innerHTML = `
    <img src="${src}" style="max-width:90vw;max-height:80vh;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,0.8)"/>
    <p style="color:#aaa;margin-top:12px;font-size:0.9rem">${label} — click để đóng</p>
  `;
  overlay.addEventListener('click', () => overlay.remove());
  document.body.appendChild(overlay);
}

// ══════════════════════════════════════════════════════════════
// 2. ADD PERSON PANEL
// ══════════════════════════════════════════════════════════════

const addFilesInput   = document.getElementById('add-files');
const addPreviewWrap  = document.getElementById('add-preview-wrap');
const addUploadLabel  = document.querySelector('#panel-add .upload-label');
const addBtn          = document.getElementById('add-btn');

addFilesInput.addEventListener('change', async () => {
  addPreviewWrap.innerHTML = '';
  const files = [...addFilesInput.files];
  if (!files.length) {
    addPreviewWrap.classList.add('hidden');
    addUploadLabel.classList.remove('hidden');
    return;
  }
  addPreviewWrap.classList.remove('hidden');
  addUploadLabel.classList.add('hidden');
  for (const f of files) {
    const url = await readFileAsDataURL(f);
    const img = document.createElement('img');
    img.src = url;
    img.className = 'preview-img';
    img.alt = f.name;
    img.title = f.name;
    addPreviewWrap.appendChild(img);
  }
  hideMessage('add-message');
});

addBtn.addEventListener('click', async () => {
  const name  = document.getElementById('add-name').value.trim();
  const files = [...addFilesInput.files];

  if (!name) { showToast('Vui lòng nhập tên người.', 'error'); return; }
  if (!files.length) { showToast('Vui lòng chọn ít nhất 1 ảnh X-ray.', 'error'); return; }

  hideMessage('add-message');
  showLoading(`🦷 Đang đăng ký "${name}"…`);

  try {
    const fd = new FormData();
    fd.append('name', name);
    files.forEach(f => fd.append('files', f));

    const resp = await fetch('/add_person', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!resp.ok) {
      showMessage('add-message', '❌ ' + (data.detail || 'Lỗi khi thêm người.'), 'error');
      return;
    }
    // Reset form
    document.getElementById('add-name').value = '';
    addFilesInput.value = '';
    addPreviewWrap.innerHTML = '';
    addPreviewWrap.classList.add('hidden');
    addUploadLabel.classList.remove('hidden');

    showMessage('add-message', '✅ ' + data.message, 'success');
    showToast(`✅ Đã thêm "${name}" thành công!`);
    refreshDeleteDropdown();

  } catch (err) {
    showMessage('add-message', '❌ Lỗi kết nối: ' + err.message, 'error');
  } finally {
    hideLoading();
  }
});

// ══════════════════════════════════════════════════════════════
// 3. DELETE PERSON PANEL
// ══════════════════════════════════════════════════════════════

const deleteSelect = document.getElementById('delete-select');
const deleteBtn    = document.getElementById('delete-btn');

async function refreshDeleteDropdown() {
  try {
    const resp = await fetch('/list_persons');
    const data = await resp.json();
    const persons = data.persons || [];

    deleteSelect.innerHTML = '<option value="">— Chọn người —</option>';
    persons.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      deleteSelect.appendChild(opt);
    });

    document.getElementById('person-count-badge').textContent = `${persons.length} người`;
    deleteBtn.disabled = !deleteSelect.value;
  } catch (e) {
    console.error('Failed to refresh persons:', e);
  }
}

deleteSelect.addEventListener('change', () => {
  deleteBtn.disabled = !deleteSelect.value;
  hideMessage('delete-message');
});

deleteBtn.addEventListener('click', async () => {
  const name = deleteSelect.value;
  if (!name) return;

  if (!confirm(`Bạn có chắc muốn xóa "${name}" khỏi database?`)) return;

  showLoading(`🗑️ Đang xóa "${name}"…`);
  try {
    const resp = await fetch(`/delete_person?name=${encodeURIComponent(name)}`, { method: 'DELETE' });
    const data = await resp.json();

    if (!resp.ok) {
      showMessage('delete-message', '❌ ' + (data.detail || 'Lỗi khi xóa.'), 'error');
      return;
    }
    showMessage('delete-message', '✅ ' + data.message, 'success');
    showToast(`🗑️ Đã xóa "${name}".`);
    await refreshDeleteDropdown();

  } catch (err) {
    showMessage('delete-message', '❌ Lỗi kết nối: ' + err.message, 'error');
  } finally {
    hideLoading();
  }
});

// ── Init ──────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Pre-load persons list on page load so delete tab is ready
refreshDeleteDropdown();
