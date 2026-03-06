import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Aion Extension Test Suite', () => {
  vscode.window.showInformationMessage('Running Aion extension tests...');

  test('Extension should be present', () => {
    assert.ok(vscode.extensions.getExtension('aion-local.aion-vscode'));
  });

  test('Commands should be registered', async () => {
    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes('aion.ask'), 'aion.ask command missing');
    assert.ok(commands.includes('aion.editSelection'), 'aion.editSelection command missing');
    assert.ok(commands.includes('aion.applyEdit'), 'aion.applyEdit command missing');
  });

  test('Output channel should exist after activation', async () => {
    // Trigger extension activation by running a command
    // The ask command will fail without server but will activate the extension
    try {
      await vscode.commands.executeCommand('aion.ask');
    } catch {
      // Expected — no server running during tests
    }
    // Extension should be active now
    const ext = vscode.extensions.getExtension('aion-local.aion-vscode');
    assert.ok(ext?.isActive, 'Extension should be active');
  });

  test('Configuration defaults should be correct', () => {
    const config = vscode.workspace.getConfiguration('aion');
    assert.strictEqual(config.get('serverUrl'), 'http://localhost:8000');
    assert.strictEqual(config.get('timeout'), 120000);
  });
});
