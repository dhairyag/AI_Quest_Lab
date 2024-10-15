document.addEventListener('DOMContentLoaded', function() {
  const notesContainer = document.getElementById('notesContainer');

  chrome.storage.local.get(null, function(items) {
    for (let [tabId, tabData] of Object.entries(items)) {
      if (tabData.notes) {
        const noteElement = document.createElement('div');
        noteElement.className = 'note';
        noteElement.innerHTML = `
          <h2>${tabData.filename || 'Unnamed Tab'} (Tab ID: ${tabId})</h2>
          <pre>${tabData.notes}</pre>
        `;
        notesContainer.appendChild(noteElement);
      }
    }

    if (notesContainer.children.length === 0) {
      notesContainer.textContent = 'No notes found.';
    }
  });
});
