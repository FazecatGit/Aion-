import { app, BrowserWindow, ipcMain, desktopCapturer, screen, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// webpack plugin magic constants, replaced at build time
declare const MAIN_WINDOW_WEBPACK_ENTRY: string;
declare const MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY: string;

let pythonProcess: ChildProcess;

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,       // Required for file.path to expose full OS path in <input type="file">
      webSecurity: false,  // Allows localhost:8000
      preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
    },
  });

  // strip or relax any CSP headers sent by the dev server so our fetches
  // aren't blocked. the header may come from webpack-dev-server and can
  // override the <meta> tag in index.html.
  mainWindow.webContents.session.webRequest.onHeadersReceived(
    (details, callback) => {
      const headers = details.responseHeaders ?? {};
      // remove CSP entirely, or you could set your own value here
      delete headers['Content-Security-Policy'];
      delete headers['content-security-policy'];
      callback({ responseHeaders: headers });
    }
  );

  // Load renderer
  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
  } else {
    mainWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY);  // webpack dev server URL
  }

  mainWindow.webContents.openDevTools();  // Auto-open console
}

function startPython() {
  const aionRoot = path.join(__dirname, '../../../../');  // Aion root
  const venvPython = path.join(aionRoot, '.venv', 'Scripts', 'python.exe');  // Windows venv

  pythonProcess = spawn(venvPython, ['-m', 'uvicorn', 'api:app', '--port', '8000', '--reload'], {
    cwd: aionRoot,
    stdio: 'pipe',
  });

  pythonProcess.stdout?.on('data', d => console.log('[Python]', d.toString()));
  pythonProcess.stderr?.on('data', d => console.error('[Python]', d.toString()));
}

app.whenReady().then(() => {
  // IPC handler: open a native file picker and return the full absolute path(s)
  ipcMain.handle('open-file-dialog', async (_event, opts: { multiple?: boolean } = {}) => {
    const result = await dialog.showOpenDialog({
      properties: opts.multiple ? ['openFile', 'multiSelections'] : ['openFile'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return opts.multiple ? result.filePaths : result.filePaths[0];
  });

  // IPC handler: capture the entire primary screen as a PNG data URL
  ipcMain.handle('capture-screen', async () => {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.size;
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width, height },
    });
    if (sources.length === 0) return null;
    // Return the first screen's thumbnail as a PNG data URL
    return sources[0].thumbnail.toDataURL();
  });

  startPython();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  pythonProcess?.kill();
});
