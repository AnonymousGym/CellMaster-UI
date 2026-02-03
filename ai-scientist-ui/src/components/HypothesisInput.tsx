import React, { ChangeEvent, useRef, useState } from 'react';
import { Box, TextField, Button, Typography, InputAdornment, IconButton, Grid } from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { useAppContext } from '../hooks/useAppContext';
import axios from 'axios';
import { usePipeline } from '../context/PipelineContext';
import Othermethod from './Othermethod';

interface FileWithPath extends File {
  path?: string;
  displayPath?: string;
}

const HypothesisInput: React.FC = () => {
  //====================
  // State management
  //====================
  const { state, setHypothesis, setOriginalGrouping, setH5adFile, setMetadataFile, setIsPipelineRunning } = useAppContext();
  const { setCurrentStage } = usePipeline();
  const h5adInputRef = useRef<HTMLInputElement>(null);
  const metadataInputRef = useRef<HTMLInputElement>(null);

  // New state for Celltypist Model input
  const [celltypistModel, setCelltypistModel] = useState('');

  //====================
  // Event handlers
  //====================
  const handleHypothesisChange = (e: ChangeEvent<HTMLInputElement>) => {
    setHypothesis(e.target.value);
  };
  
  const handleOriginalGroupingChange = (e: ChangeEvent<HTMLInputElement>) => {
    setOriginalGrouping(e.target.value);
  };

  const handleCelltypistModelChange = (e: ChangeEvent<HTMLInputElement>) => {
    setCelltypistModel(e.target.value);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>, fileType: 'h5ad' | 'metadata') => {
    const file = e.target.files?.[0] || null;
    if (fileType === 'h5ad') {
      setH5adFile(file);
    } else {
      setMetadataFile(file);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>, fileType: 'h5ad' | 'metadata') => {
    const filepath = e.target.value;
    
    if (filepath) {
      const fileWithPath = new File([], filepath) as FileWithPath;
      fileWithPath.path = filepath;
      fileWithPath.displayPath = filepath;
      
      if (fileType === 'h5ad') {
        setH5adFile(fileWithPath);
      } else {
        setMetadataFile(fileWithPath);
      }
    } else {
      if (fileType === 'h5ad') {
        setH5adFile(null);
      } else {
        setMetadataFile(null);
      }
    }
  };

  const triggerFileInput = (inputRef: React.RefObject<HTMLInputElement>) => {
    inputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!state.h5adFile) {
      return;
    }

    setIsPipelineRunning(true);

    const formData = new FormData();
    
    const h5adFile = state.h5adFile as FileWithPath;
    if (h5adFile.path) {
      formData.append('h5ad_path', h5adFile.path);
    } else {
      formData.append('h5ad', state.h5adFile);
    }

    formData.append("original_grouping", state.original_grouping);
    
    if (state.metadataFile) {
      const metadataFile = state.metadataFile as FileWithPath;
      if (metadataFile.path) {
        formData.append('csv_path', metadataFile.path);
      } else {
        formData.append('csv', state.metadataFile);
      }
    }
    
    formData.append('celltypistModel', celltypistModel);

    try {
      await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setCurrentStage('hypothesis');
    } catch (error) {
      console.error('Upload error:', error);
      setIsPipelineRunning(false);
    }
  };

  //====================
  // Render
  //====================
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>Input Data</Typography>
      
      {/* Iteration and Original Grouping side by side */}
      <Grid container spacing={3} alignItems="center">
        <Grid item>
          <TextField
            label="Iteration"
            type="number"
            value={state.currentIteration}
            InputProps={{
              readOnly: true,
            }}
            sx={{ width: '100px' }}
          />
        </Grid>
        <Grid item>
          <TextField
            label="Original Grouping"
            value={state.original_grouping}
            onChange={handleOriginalGroupingChange}
            placeholder="leiden? seurat_clusters?"
            InputProps={{
              readOnly: state.isPipelineRunning,
            }}
            sx={{ width: '200px' }}  // Adjust the width as needed
          />
        </Grid>
        <Grid item>
          <TextField
            fullWidth
            label="Celltypist Model"
            value={celltypistModel}
            onChange={handleCelltypistModelChange}
            placeholder="Enter Celltypist model"
            InputProps={{
              readOnly: state.isPipelineRunning,
            }}
          />
        </Grid>
        <Grid item>
          <Othermethod />
        </Grid>
      </Grid>

      <input
        type="file"
        ref={h5adInputRef}
        onChange={(e: ChangeEvent<HTMLInputElement>) => handleFileChange(e, 'h5ad')}
        style={{ display: 'none' }}
        accept=".h5ad"
      />
      <TextField
        fullWidth
        label="H5AD File: Enter file path or click folder icon to browse"
        variant="outlined"
        value={state.h5adFile?.name || ''}
        onChange={(e) => handleInputChange(e, 'h5ad')}
        placeholder="Enter file path or click folder icon to browse"
        margin="normal"
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <IconButton
                onClick={(e) => {
                  e.stopPropagation();
                  triggerFileInput(h5adInputRef);
                }}
                disabled={state.isPipelineRunning}
              >
                <FolderOpenIcon />
              </IconButton>
            </InputAdornment>
          ),
        }}
      />

      <input
        type="file"
        ref={metadataInputRef}
        onChange={(e: ChangeEvent<HTMLInputElement>) => handleFileChange(e, 'metadata')}
        style={{ display: 'none' }}
        accept=".csv,.tsv,.txt"
      />
      <TextField
        fullWidth
        label="Metadata File: Enter file path or click folder icon to browse"
        variant="outlined"
        value={state.metadataFile?.name || ''}
        onChange={(e) => handleInputChange(e, 'metadata')}
        placeholder="Enter file path or click folder icon to browse"
        margin="normal"
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <IconButton
                onClick={(e) => {
                  e.stopPropagation();
                  triggerFileInput(metadataInputRef);
                }}
                disabled={state.isPipelineRunning}
              >
                <FolderOpenIcon />
              </IconButton>
            </InputAdornment>
          ),
        }}
      />

      <TextField
        fullWidth
        variant="outlined"
        multiline
        rows={4}
        value={state.hypothesis}
        onChange={handleHypothesisChange}
        placeholder="Enter your hypothesis..."
        margin="normal"
        sx={{ flexGrow: 1 }}
        disabled={state.isPipelineRunning}
      />

      <Button
        fullWidth
        variant="contained"
        color="primary"
        sx={{ mt: 2 }}
        disabled={state.isPipelineRunning || !state.h5adFile || !state.hypothesis || !state.original_grouping}
        onClick={handleUpload}
      >
        EXPLORE CLUSTER
      </Button>
    </Box>
  );
};

export default HypothesisInput;