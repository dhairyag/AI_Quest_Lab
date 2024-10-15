let isSelecting = false;

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "selectText") {
    isSelecting = true;
    document.body.classList.add('mn-selecting');
    document.addEventListener('mouseup', handleMouseUp);
  }
});

function handleMouseUp() {
  if (isSelecting) {
    const selectedText = window.getSelection().toString();
    if (selectedText) {
      showFormatOptions(selectedText);
    }
    isSelecting = false;
    document.body.classList.remove('mn-selecting');
    document.removeEventListener('mouseup', handleMouseUp);
  }
}

function showFormatOptions(text) {
  const options = [
    {name: 'Normal', prefix: ''},
    {name: 'Bullet List (-)', prefix: '- '},
    {name: 'Heading 1 (H1)', prefix: '# '},
    {name: 'Heading 2 (H2)', prefix: '## '},
    {name: 'Heading 3 (H3)', prefix: '### '},
    {name: 'Heading 4 (H4)', prefix: '#### '},
    {name: 'Numbered List (1.)', prefix: '1. '},
    {name: 'Blockquote (>)', prefix: '> '}
  ];

  const dialog = document.createElement('div');
  dialog.style.position = 'fixed';
  dialog.style.top = '20px';
  dialog.style.right = '20px';
  dialog.style.backgroundColor = 'white';
  dialog.style.border = '1px solid black';
  dialog.style.padding = '10px';
  dialog.style.zIndex = '9999';
  dialog.style.display = 'flex';
  dialog.style.flexDirection = 'column';

  options.forEach(option => {
    const button = document.createElement('button');
    button.textContent = option.name;
    button.style.margin = '5px 0';
    button.addEventListener('click', () => {
      saveNote(text, option.prefix);
      document.body.removeChild(dialog);
    });
    dialog.appendChild(button);
  });

  document.body.appendChild(dialog);
}

function saveNote(text, prefix) {
  const formattedText = prefix + text + '\n\n';
  chrome.runtime.sendMessage({action: "saveNote", text: formattedText});
}

// Add custom cursor style
const style = document.createElement('style');
style.textContent = `
  .mn-selecting {
    cursor: text !important;
  }
  .mn-selecting::after {
    content: '';
    display: inline-block;
    width: 1px;
    height: 1em;
    background-color: black;
    animation: blink 0.7s infinite;
    position: absolute;
    pointer-events: none;
  }
  @keyframes blink {
    0% { opacity: 0; }
    50% { opacity: 1; }
    100% { opacity: 0; }
  }
`;
document.head.appendChild(style);
