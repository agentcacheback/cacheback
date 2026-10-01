if (typeof document !== 'undefined') {
  const byId = (id) => document.getElementById(id);
  byId('cite-link').addEventListener('click', () => { byId('citation').open = true; });
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
    button.addEventListener('click', async () => {
      const content = byId(button.dataset.copy);
      try {
        await navigator.clipboard.writeText(content.textContent);
        button.textContent = 'Copied';
        byId('copy-status').textContent = 'Copied to clipboard.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(content);
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = 'Text selected';
        byId('copy-status').textContent = 'Clipboard unavailable. Text selected; use your device’s copy command.';
      }
    });
  });
}
