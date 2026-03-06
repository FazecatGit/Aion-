import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  captureScreen: () => ipcRenderer.invoke('capture-screen'),
  openFileDialog: (opts?: { multiple?: boolean }) => ipcRenderer.invoke('open-file-dialog', opts),
});