"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const assert = __importStar(require("assert"));
const vscode = __importStar(require("vscode"));
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
        }
        catch {
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
//# sourceMappingURL=extension.test.js.map