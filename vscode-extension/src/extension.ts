/**
 * Aion VSCode Extension
 *
 * Connects to Aion's localhost FastAPI backend for:
 *   - RAG-powered knowledge queries
 *   - Agent-based inline code edits
 *   - Apply/reject workflow with diff preview
 *
 * Fully offline — communicates only with localhost.
 */

import * as vscode from "vscode";

// ── Aion Client ─────────────────────────────────────────────────────────────

class AionClient {
  private baseUrl: string;
  private timeout: number;

  constructor() {
    const config = vscode.workspace.getConfiguration("aion");
    this.baseUrl = config.get<string>("serverUrl", "http://localhost:8000");
    this.timeout = config.get<number>("timeout", 120_000);
  }

  /** Refresh config (call when settings change). */
  reload(): void {
    const config = vscode.workspace.getConfiguration("aion");
    this.baseUrl = config.get<string>("serverUrl", "http://localhost:8000");
    this.timeout = config.get<number>("timeout", 120_000);
  }

  /** POST JSON to an Aion endpoint. */
  async post(endpoint: string, body: Record<string, unknown>): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      return await resp.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // ── High-level helpers ──────────────────────────────────────────────────

  async query(question: string, sessionId?: string) {
    return this.post("/query", { question, session_id: sessionId });
  }

  async agentEdit(filePath: string, instruction: string, sessionId?: string) {
    return this.post("/agent/edit", {
      file_path: filePath,
      instruction,
      session_id: sessionId,
    });
  }

  async agentApply(
    filePath: string,
    instruction: string,
    newSource?: string,
    sessionId?: string
  ) {
    return this.post("/agent/apply", {
      file_path: filePath,
      instruction,
      new_source: newSource,
      session_id: sessionId,
    });
  }
}

// ── State ───────────────────────────────────────────────────────────────────

let client: AionClient;

interface PendingEdit {
  filePath: string;
  instruction: string;
  newSource: string;
  diff: string;
  explanation: string;
}

let pendingEdit: PendingEdit | undefined;

// ── Output channel ──────────────────────────────────────────────────────────

let outputChannel: vscode.OutputChannel;

function log(msg: string): void {
  outputChannel.appendLine(`[Aion] ${msg}`);
}

// ── Commands ────────────────────────────────────────────────────────────────

/** Ask a RAG question via the input box. */
async function askCommand(): Promise<void> {
  const question = await vscode.window.showInputBox({
    prompt: "Ask Aion a question (RAG-powered)",
    placeHolder: "e.g. How does the router select a pipeline?",
  });
  if (!question) {
    return;
  }

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Aion: Thinking…" },
    async () => {
      try {
        const data = await client.query(question);
        if (data.error) {
          vscode.window.showErrorMessage(`Aion error: ${data.error}`);
          return;
        }

        // Show answer in output channel
        outputChannel.show(true);
        outputChannel.appendLine("─".repeat(60));
        outputChannel.appendLine(`Q: ${question}`);
        outputChannel.appendLine("");
        outputChannel.appendLine(data.answer ?? data.result ?? JSON.stringify(data));
        if (data.citations?.length) {
          outputChannel.appendLine("");
          outputChannel.appendLine("Sources:");
          for (const c of data.citations) {
            outputChannel.appendLine(`  • ${c}`);
          }
        }
        outputChannel.appendLine("");
      } catch (err: any) {
        vscode.window.showErrorMessage(
          `Aion: Could not reach server. Is it running?\n${err.message}`
        );
      }
    }
  );
}

/** Edit selected code via the agent. */
async function editSelectionCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("Aion: No active editor.");
    return;
  }

  const filePath = editor.document.uri.fsPath;
  const instruction = await vscode.window.showInputBox({
    prompt: "What should Aion do to this file?",
    placeHolder: "e.g. Add error handling to the parse function",
  });
  if (!instruction) {
    return;
  }

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Aion Agent: Working…" },
    async () => {
      try {
        const data = await client.agentEdit(filePath, instruction);

        if (data.error || data.status === "error") {
          vscode.window.showErrorMessage(`Aion: ${data.error ?? data.explanation}`);
          return;
        }

        if (data.status === "no_changes") {
          vscode.window.showInformationMessage("Aion: No changes needed.");
          return;
        }

        // Store pending edit for Apply/Reject
        pendingEdit = {
          filePath,
          instruction,
          newSource: data.new_source ?? "",
          diff: data.diff ?? "",
          explanation: data.explanation ?? "",
        };

        // Show diff in output
        outputChannel.show(true);
        outputChannel.appendLine("═".repeat(60));
        outputChannel.appendLine(`Agent edit: ${filePath}`);
        outputChannel.appendLine(`Instruction: ${instruction}`);
        outputChannel.appendLine("─".repeat(60));
        if (data.diff) {
          outputChannel.appendLine(data.diff);
        }
        outputChannel.appendLine("─".repeat(60));
        outputChannel.appendLine(data.explanation ?? "");
        outputChannel.appendLine("");

        // Prompt user
        const action = await vscode.window.showInformationMessage(
          "Aion: Changes ready. Apply them?",
          "Apply",
          "Show Diff",
          "Reject"
        );

        if (action === "Apply") {
          await applyEditCommand();
        } else if (action === "Show Diff") {
          await showDiff();
        } else {
          pendingEdit = undefined;
          vscode.window.showInformationMessage("Aion: Changes rejected.");
        }
      } catch (err: any) {
        vscode.window.showErrorMessage(
          `Aion: Could not reach server.\n${err.message}`
        );
      }
    }
  );
}

/** Apply the pending agent edit. */
async function applyEditCommand(): Promise<void> {
  if (!pendingEdit) {
    vscode.window.showWarningMessage("Aion: No pending edit to apply.");
    return;
  }

  try {
    const data = await client.agentApply(
      pendingEdit.filePath,
      pendingEdit.instruction,
      pendingEdit.newSource
    );

    if (data.error || data.status === "error") {
      vscode.window.showErrorMessage(`Aion apply failed: ${data.error}`);
      return;
    }

    // Reload the file in the editor
    const doc = vscode.workspace.textDocuments.find(
      (d) => d.uri.fsPath === pendingEdit!.filePath
    );
    if (doc) {
      // Revert to reload from disk
      await vscode.commands.executeCommand("workbench.action.files.revert");
    }

    vscode.window.showInformationMessage("Aion: Changes applied.");
    log(`Applied edit to ${pendingEdit.filePath}`);
    pendingEdit = undefined;
  } catch (err: any) {
    vscode.window.showErrorMessage(`Aion apply error: ${err.message}`);
  }
}

/** Show a diff of the pending edit. */
async function showDiff(): Promise<void> {
  if (!pendingEdit) {
    return;
  }

  // Create a virtual document with the new content
  const originalUri = vscode.Uri.file(pendingEdit.filePath);
  const newUri = vscode.Uri.parse(
    `untitled:${pendingEdit.filePath}.aion-preview`
  );

  // Open the new source as an untitled doc
  const doc = await vscode.workspace.openTextDocument(newUri);
  const editor = await vscode.window.showTextDocument(doc, { preview: true });
  await editor.edit((edit) => {
    edit.insert(new vscode.Position(0, 0), pendingEdit!.newSource);
  });

  // Show diff between original and new
  await vscode.commands.executeCommand(
    "vscode.diff",
    originalUri,
    newUri,
    `Aion: ${pendingEdit.instruction}`
  );
}

// ── Extension lifecycle ─────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("Aion");
  client = new AionClient();

  log("Extension activated");

  // Reload config on settings change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("aion")) {
        client.reload();
        log("Settings reloaded");
      }
    })
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("aion.ask", askCommand),
    vscode.commands.registerCommand("aion.editSelection", editSelectionCommand),
    vscode.commands.registerCommand("aion.applyEdit", applyEditCommand)
  );

  log("Commands registered: aion.ask, aion.editSelection, aion.applyEdit");
}

export function deactivate(): void {
  outputChannel?.dispose();
}
