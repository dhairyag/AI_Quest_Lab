document.addEventListener('DOMContentLoaded', function() {
  const selectTextBtn = document.getElementById('selectText');
  const viewNotesBtn = document.getElementById('viewNotes');
  const downloadNotesBtn = document.getElementById('downloadNotes');
  const clearAllNotesBtn = document.getElementById('clearAllNotes');
  const filenameInput = document.getElementById('filename');
  const setFilenameBtn = document.getElementById('setFilename');

  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    const tabId = tabs[0].id.toString();
    chrome.storage.local.get(tabId, function(result) {
      if (result[tabId] && result[tabId].filename) {
        filenameInput.value = result[tabId].filename;
      }
    });
  });

  setFilenameBtn.addEventListener('click', function() {
    const filename = filenameInput.value.trim();
    if (filename) {
      chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const tabId = tabs[0].id.toString();
        chrome.storage.local.get(tabId, function(result) {
          const tabData = result[tabId] || {};
          tabData.filename = filename;
          chrome.storage.local.set({[tabId]: tabData});
        });
      });
    }
  });

  selectTextBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {action: "selectText"});
      window.close();
    });
  });

  viewNotesBtn.addEventListener('click', function() {
    chrome.tabs.create({url: 'view-notes.html'});
  });

  downloadNotesBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const tabId = tabs[0].id.toString();
      chrome.storage.local.get(tabId, function(result) {
        const tabData = result[tabId] || {};
        const notes = tabData.notes || '';
        const filename = tabData.filename || 'notes';
        const blob = new Blob([notes], {type: 'text/markdown'});
        const url = URL.createObjectURL(blob);
        chrome.downloads.download({
          url: url,
          filename: `${filename}.md`,
          saveAs: true
        });
      });
    });
  });

  clearAllNotesBtn.addEventListener('click', function() {
    if (confirm('Are you sure you want to clear all notes from all tabs?')) {
      chrome.storage.local.clear(function() {
        alert('All notes have been cleared.');
      });
    }
  });
});
