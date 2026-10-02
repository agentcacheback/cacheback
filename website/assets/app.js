if (typeof document !== 'undefined') {
  const byId = (id) => document.getElementById(id);
  const video = byId('coding-video');
  const play = byId('play-demo');
  play.hidden = false;
  play.addEventListener('click', async () => {
    try {
      await video.play();
      video.focus();
    } catch {
      byId('copy-status').textContent = 'Playback unavailable. Try the Open in full window link below the video.';
    }
  });
  video.addEventListener('play', () => { play.hidden = true; });

  document.querySelectorAll('[data-copy]').forEach((button) => {
    let reset;
    button.addEventListener('click', async () => {
      const content = byId(button.dataset.copy);
      try {
        await navigator.clipboard.writeText(content.textContent);
        button.querySelector('.copy-feedback').textContent = 'Copied';
        byId('copy-status').textContent = 'Copied to clipboard.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(content);
        selection.removeAllRanges();
        selection.addRange(range);
        button.querySelector('.copy-feedback').textContent = 'Selected';
        byId('copy-status').textContent = 'Clipboard unavailable. Text selected; use your device’s copy command.';
      }
      clearTimeout(reset);
      button.classList.add('is-copied');
      reset = setTimeout(() => button.classList.remove('is-copied'), 1800);
    });
  });
}
