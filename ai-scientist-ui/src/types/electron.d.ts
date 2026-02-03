export {};

declare global {
  interface Window {
    electron: {
      openFileDialog: (options: Electron.OpenDialogOptions) => Promise<Electron.OpenDialogReturnValue>;
      processFile: (filePath: string, fileType: 'h5ad' | 'metadata') => Promise<{
        file: {
          name: string;
          path: string;
          lastModified: number;
          size: number;
        };
        destinationPath: string;
      }>;
    };
  }
}