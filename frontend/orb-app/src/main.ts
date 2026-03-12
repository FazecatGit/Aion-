import { app, BrowserWindow, ipcMain, desktopCapturer, screen, dialog, globalShortcut } from 'electron';
import { spawn, execSync, ChildProcess } from 'child_process';
import path from 'path';

// webpack plugin magic constants, replaced at build time
declare const MAIN_WINDOW_WEBPACK_ENTRY: string;
declare const MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY: string;

let pythonProcess: ChildProcess;
let mainWindow: BrowserWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
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

  // IPC handler: capture screen with area selection overlay
  ipcMain.handle('capture-screen', async () => {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.size;
    const scaleFactor = primaryDisplay.scaleFactor;

    // Step 1: Take a full screenshot of the entire screen
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: width * scaleFactor, height: height * scaleFactor },
    });
    if (sources.length === 0) return null;
    const fullScreenshot = sources[0].thumbnail;

    // Step 2: Open a frameless overlay window showing the screenshot,
    //         let the user drag-select a region
    return new Promise<string | null>((resolve) => {
      const overlay = new BrowserWindow({
        fullscreen: true,
        frame: false,
        transparent: false,
        alwaysOnTop: true,
        skipTaskbar: true,
        resizable: false,
        webPreferences: {
          nodeIntegration: false,
          contextIsolation: true,
          preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
        },
      });

      const imgDataUrl = fullScreenshot.toDataURL();

      // Inject the selection UI directly via loadURL with a data URL
      const html = `<!DOCTYPE html>
<html><head><style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { cursor:crosshair; overflow:hidden; user-select:none; }
  canvas { display:block; }
  #sel { position:fixed; border:2px solid #5533ff; background:rgba(85,51,255,0.15); display:none; pointer-events:none; z-index:10; }
  #hint { position:fixed; top:16px; left:50%; transform:translateX(-50%); color:#fff; background:rgba(0,0,0,0.7); padding:8px 18px; border-radius:8px; font:14px sans-serif; z-index:20; }
</style></head><body>
<div id="hint">Click and drag to select area. Press Escape to cancel.</div>
<div id="sel"></div>
<canvas id="c"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const sel = document.getElementById('sel');
const img = new Image();
img.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  // dim the screen slightly
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};
img.src = ${JSON.stringify(imgDataUrl)};

let sx=0, sy=0, dragging=false;
document.addEventListener('mousedown', e => {
  sx = e.clientX; sy = e.clientY;
  dragging = true;
  sel.style.display = 'block';
  sel.style.left = sx+'px'; sel.style.top = sy+'px';
  sel.style.width = '0'; sel.style.height = '0';
});
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const x = Math.min(sx, e.clientX), y = Math.min(sy, e.clientY);
  const w = Math.abs(e.clientX - sx), h = Math.abs(e.clientY - sy);
  sel.style.left = x+'px'; sel.style.top = y+'px';
  sel.style.width = w+'px'; sel.style.height = h+'px';
});
document.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  const x = Math.min(sx, e.clientX), y = Math.min(sy, e.clientY);
  const w = Math.abs(e.clientX - sx), h = Math.abs(e.clientY - sy);
  if (w < 10 || h < 10) {
    // too small — cancel
    window.electronAPI.sendCropResult(null);
    return;
  }
  // scale coords from screen CSS pixels to image pixels
  const scaleX = img.naturalWidth / canvas.width;
  const scaleY = img.naturalHeight / canvas.height;
  window.electronAPI.sendCropResult({
    x: Math.round(x * scaleX),
    y: Math.round(y * scaleY),
    w: Math.round(w * scaleX),
    h: Math.round(h * scaleY),
  });
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') window.electronAPI.sendCropResult(null);
});
</script></body></html>`;

      overlay.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);

      // Listen for the crop result from the overlay
      ipcMain.once('crop-result', (_event, rect: { x: number; y: number; w: number; h: number } | null) => {
        overlay.close();
        if (!rect) {
          resolve(null);
          return;
        }
        // Crop the original full screenshot
        const cropped = fullScreenshot.crop({
          x: rect.x, y: rect.y,
          width: rect.w, height: rect.h,
        });
        resolve(cropped.toDataURL());
      });

      overlay.on('closed', () => {
        // If overlay was closed without sending, resolve null
        resolve(null);
      });
    });
  });

  startPython();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  if (pythonProcess?.pid) {
    // On Windows, process.kill() only kills the parent — the uvicorn --reload
    // worker child survives as an orphan and keeps holding port 8000.
    // taskkill /T kills the entire process tree.
    try {
      execSync(`taskkill /PID ${pythonProcess.pid} /T /F`);
    } catch {
      pythonProcess.kill();
    }
  }
});
