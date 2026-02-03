import React, { useCallback } from 'react';
import { AppBar, Button, Toolbar, Typography } from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import { useAppContext } from '../hooks/useAppContext';
import JSZip from 'jszip';

const BASE_URL = 'http://localhost:5000/';

const Header: React.FC = () => {
  //====================
  // State management
  //====================
  const { state, resetApplication } = useAppContext();
  const { analysisResults } = state;

  //====================
  // Derived state
  //====================
  const isExportEnabled = analysisResults?.[analysisResults.length - 1]?.h5ad;

  //====================
  // Event handlers
  //====================
  const handleDownload = useCallback(async () => {
    const filename = analysisResults?.[analysisResults.length - 1]?.h5ad;
    if (!filename) return;

    const zip = new JSZip();

    // Add h5ad file
    const h5adFileUrl = BASE_URL + 'h5ad/' + filename;
    const h5adResponse = await fetch(h5adFileUrl);
    const h5adBlob = await h5adResponse.blob();
    zip.file("annotated_" + filename, h5adBlob);

    // Add history text content
    const historyUrl = BASE_URL + 'history';
    const historyResponse = await fetch(historyUrl);
    const historyText = await historyResponse.text();
    zip.file("history.txt", historyText);

    // Add umap and dotplot images
    const imageTypes = ['umap', 'dotplot'];
    for (const type of imageTypes) {
      const imageListUrl = BASE_URL + type + '/list';
      const imageListResponse = await fetch(imageListUrl);
      const imageList = await imageListResponse.json();
      const folder = zip.folder(type);
      for (const img of imageList) {
        const imgUrl = BASE_URL + type + '/' + img;
        const imgResponse = await fetch(imgUrl);
        const imgBlob = await imgResponse.blob();
        folder?.file(img, imgBlob);
      }
    }

    // Generate and download ZIP file
    const content = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(content);
    link.download = "analysis_results.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [analysisResults]);

  const handleReset = useCallback(() => {
    if (window.confirm('Are you sure you want to reset the application? All progress will be lost.')) {
      resetApplication();
    }
  }, [resetApplication]);

  //====================
  // Render
  //====================
  return (
    <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: (theme) => `1px solid ${theme.palette.divider}` }}>
      <Toolbar sx={{ flexWrap: 'wrap' }}>
        <Typography variant="h6" color="inherit" noWrap sx={{ flexGrow: 1 }}>
          AI Scientist
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RestartAltIcon/>}
          onClick={handleReset}
          sx={{ my: 1, mx: 1.5 }}
        >
          Reset
        </Button>
        <Button
          variant="outlined"
          startIcon={<FileDownloadIcon/>}
          disabled={!isExportEnabled}
          onClick={handleDownload}
          sx={{ my: 1, mx: 1.5 }}
        >
          Export
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default Header;