# Aion VSCode Extension

Local AI coding agent — queries, inline edits, and diff preview, all offline.

## Setup

```bash
cd vscode-extension
npm install
npm run compile
```

## Install locally

```bash
npm run package          # produces aion-vscode-0.1.0.vsix
code --install-extension aion-vscode-0.1.0.vsix
```

## Usage

| Command                    | Keybinding       | Description                               |
|----------------------------|------------------|-------------------------------------------|
| **Aion: Ask a Question**   | `Ctrl+Shift+A`   | RAG-powered knowledge query               |
| **Aion: Edit Selected Code** | `Ctrl+Shift+E` | Agent edits the active file               |
| **Aion: Apply Pending Edit** | —              | Apply the last proposed edit              |

## Configuration

| Setting          | Default                   | Description               |
|------------------|---------------------------|---------------------------|
| `aion.serverUrl` | `http://localhost:8000`   | Aion backend URL          |
| `aion.timeout`   | `120000`                  | Request timeout (ms)      |

## Requirements

- Aion backend running (`python api.py`)
- Works fully offline — no external API calls
