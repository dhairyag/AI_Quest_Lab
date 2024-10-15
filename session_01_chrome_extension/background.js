chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "saveNote") {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const tabId = tabs[0].id.toString();
      chrome.storage.local.get(tabId, function(result) {
        let tabData = result[tabId] || {};
        tabData.notes = (tabData.notes || '') + request.text;
        chrome.storage.local.set({[tabId]: tabData});
      });
    });
  }
});
