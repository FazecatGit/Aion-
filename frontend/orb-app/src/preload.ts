import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  captureScreen: () => ipcRenderer.invoke('capture-screen'),
  sendCropResult: (rect: { x: number; y: number; w: number; h: number } | null) => ipcRenderer.send('crop-result', rect),
  openFileDialog: (opts?: { multiple?: boolean }) => ipcRenderer.invoke('open-file-dialog', opts),
});