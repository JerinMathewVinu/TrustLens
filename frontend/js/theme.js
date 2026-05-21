/**
 * TrustLens — Global Theme Toggle
 * Inject this script AFTER the nav markup on every page.
 * It reads/saves preference from localStorage (default: dark).
 */
(function () {
  const html = document.documentElement;

  /* ── Inject toggle button into nav-links ── */
  const navLinks = document.querySelector('.nav-links');
  if (navLinks) {
    const li = document.createElement('li');
    li.innerHTML = `
      <button id="themeToggle" aria-label="Toggle theme" style="
        display:inline-flex;align-items:center;gap:7px;
        font-family:'Space Mono',monospace;font-size:0.72rem;
        letter-spacing:0.08em;text-transform:uppercase;
        color:var(--muted);background:var(--surface,rgba(255,255,255,0.04));
        border:1px solid var(--border,rgba(184,255,60,0.15));
        padding:7px 14px;border-radius:4px;cursor:pointer;
        transition:all 0.25s;outline:none;white-space:nowrap;
      ">
        <span id="themeIcon" style="font-size:1rem;transition:transform 0.4s;display:inline-block;">🌙</span>
        <span id="themeLabel">Dark</span>
      </button>`;
    navLinks.appendChild(li);
  }

  const btn   = document.getElementById('themeToggle');
  const icon  = document.getElementById('themeIcon');
  const label = document.getElementById('themeLabel');

  /* ── Apply theme ── */
  function applyTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('tl-theme', theme);
    if (theme === 'light') {
      if (icon)  { icon.textContent = '☀️'; icon.style.transform = 'rotate(180deg)'; }
      if (label) label.textContent = 'Light';
      if (btn) {
        btn.style.color = 'var(--muted)';
      }
    } else {
      if (icon)  { icon.textContent = '🌙'; icon.style.transform = 'rotate(0deg)'; }
      if (label) label.textContent = 'Dark';
    }
  }

  /* ── Load saved preference (default dark) ── */
  const saved = localStorage.getItem('tl-theme') || 'dark';
  applyTheme(saved);

  /* ── Toggle on click ── */
  if (btn) {
    btn.addEventListener('click', () => {
      const current = html.getAttribute('data-theme') || 'dark';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });

    /* hover styles via JS (avoids needing extra CSS) */
    btn.addEventListener('mouseenter', () => {
      btn.style.color = 'var(--acid)';
      btn.style.borderColor = 'rgba(184,255,60,0.4)';
      btn.style.background = 'rgba(184,255,60,0.06)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.color = 'var(--muted)';
      btn.style.borderColor = 'var(--border,rgba(184,255,60,0.15))';
      btn.style.background = 'var(--surface,rgba(255,255,255,0.04))';
    });
  }
})();
