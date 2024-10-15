# Markdown Notes Chrome Extension

## Description

Markdown Notes is a Chrome extension that allows users to easily take notes from web pages and save them in Markdown format. This extension is perfect for researchers, students, or anyone who frequently needs to collect and organize information from the web. This extension is created with the help of GenAI coding tools. 

## Features

- Select and save text from any web page
- Format saved text as Markdown (normal text, bullet lists, headings, numbered lists, blockquotes)
- Organize notes by tab
- View all saved notes
- Download notes as Markdown files
- Clear all notes

## Installation

1. Clone this repository or download the source code.
2. Open Google Chrome and navigate to `chrome://extensions`.
3. Enable "Developer mode" in the top right corner.
4. Click "Load unpacked" and select the directory containing the extension files.
5. The Markdown Notes extension should now appear in your Chrome toolbar.

## Usage

### Setting a Filename

1. Click on the extension icon in the Chrome toolbar to open the popup.
2. Enter a filename in the input field at the top of the popup.
3. Click the "OK" button to save the filename for the current tab.

### Selecting and Saving Text

1. Click on the extension icon to open the popup.
2. Click the "Select Text" button.
3. The cursor will change to indicate selection mode.
4. Select the desired text on the web page.
5. After selecting, a formatting options dialog will appear.
6. Choose the desired Markdown format for the selected text:
   - Normal
   - Bullet List (-)
   - Heading 1 (H1)
   - Heading 2 (H2)
   - Heading 3 (H3)
   - Heading 4 (H4)
   - Numbered List (1.)
   - Blockquote (>)
7. The formatted text will be saved to the notes for the current tab.

### Viewing Notes

1. Click on the extension icon to open the popup.
2. Click the "View Current Notes" button.
3. A new tab will open, displaying all saved notes organized by tab.

### Downloading Notes

1. Click on the extension icon to open the popup.
2. Click the "Download Notes" button.
3. Choose a location to save the Markdown file.
4. The notes for the current tab will be downloaded as a .md file.

### Clearing All Notes

1. Click on the extension icon to open the popup.
2. Click the "Clear All Notes" button.
3. Confirm the action in the dialog box.
4. All saved notes across all tabs will be deleted.

## File Structure

- `manifest.json`: Extension configuration file
- `popup.html`: HTML for the extension popup
- `popup.js`: JavaScript for the extension popup functionality
- `content.js`: Content script for interacting with web pages
- `background.js`: Background script for handling extension events
- `view-notes.html`: HTML for the notes viewing page
- `view-notes.js`: JavaScript for displaying saved notes
- `icon16.png`, `icon48.png`, `icon128.png`: Extension icons



