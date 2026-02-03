import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, Paper, Button, CircularProgress, Checkbox, FormControlLabel, Snackbar, Alert, LinearProgress, IconButton, Slider } from '@mui/material';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import { useAppContext } from '../hooks/useAppContext';
import { usePipeline } from '../context/PipelineContext';
import axios from 'axios';

const UMAPPlot: React.FC = () => {
  //====================
  // State management
  //====================
  const { state, addUmapData } = useAppContext();
  const [selectedIteration, setSelectedIteration] = useState(1);
  const [currentIterationUmapIndex, setCurrentIterationUmapIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [localSelectedClusters, setLocalSelectedClusters] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [checkboxKey] = useState(0);
  const [isCheckboxSelected, setIsCheckboxSelected] = useState(false);
  const [isZoomedIn, setIsZoomedIn] = useState(false);
  const [isZoomingIn, setIsZoomingIn] = useState(false);
  const [isZoomingOut, setIsZoomingOut] = useState(false);
  const [clusterInfo, setClusterInfo] = useState<Record<string, number>>({});
  const [isUmapGenerated, setIsUmapGenerated] = useState(false);
  const [resolution, setResolution] = useState<number>(0.1);

  //====================
  // Derived state
  //====================
  const { analysisResults, umapData } = state;
  const currentIterationUmaps = umapData.filter(umap => umap.iteration === selectedIteration);
  const currentUmapData = currentIterationUmaps[currentIterationUmapIndex];
  const isLastUmap = selectedIteration === state.currentIteration && currentIterationUmapIndex === currentIterationUmaps.length - 1;
  const isNextIterationAvailable = umapData.some(umap => umap.iteration > selectedIteration);
  const isPrevIterationAvailable = selectedIteration > 1;

  //====================
  // Image loading
  //====================
  const loadImage = useCallback((url: string) => {
    setIsLoading(true);
    setError(null);
    const img = new Image();
    img.onload = () => {
      setIsLoading(false);
      setRetryCount(0);
    };
    img.onerror = () => {
      if (retryCount < 3) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          loadImage(url);
        }, 1000);
      } else {
        setIsLoading(false);
        setError(`Failed to load image after ${retryCount} retries`);
      }
    };
    img.src = url;
  }, [retryCount]);

  //====================
  // Effects
  //====================
  useEffect(() => {
    const lastIterationWithUmap = [...umapData]
      .reverse()
      .find(umap => umap.iteration <= state.currentIteration)?.iteration;
    
    if (lastIterationWithUmap) {
      setSelectedIteration(lastIterationWithUmap);
      const umapsForLastIteration = umapData.filter(umap => umap.iteration === lastIterationWithUmap);
      setCurrentIterationUmapIndex(umapsForLastIteration.length - 1);
    } else if (state.currentIteration > 0) {
      // If there's no UMAP for the current iteration yet, set to the current iteration
      setSelectedIteration(state.currentIteration);
      setCurrentIterationUmapIndex(0);
    }
  }, [umapData, state.currentIteration]);

  useEffect(() => {
    if (currentUmapData) {
      const fullUrl = currentUmapData.url.startsWith('http') 
        ? currentUmapData.url 
        : `http://localhost:5000/umap/${currentUmapData.url}`;
      loadImage(fullUrl);
    }
  }, [currentUmapData, loadImage]);

  useEffect(() => {
    setIsLoading(true);
    setLocalSelectedClusters([]);
    setIsCheckboxSelected(false);
  }, [selectedIteration, umapData, clusterInfo]);

  useEffect(() => {
    setIsUmapGenerated(umapData.some(umap => umap.iteration === state.currentIteration));
  }, [umapData, state.currentIteration]);

  //====================

  // Event handlers
  //====================
  const handleClusterSelect = (clusterId: string) => {
    setLocalSelectedClusters(prev => {
      const newSelectedClusters = prev.includes(clusterId) 
        ? prev.filter(id => id !== clusterId) 
        : [...prev, clusterId];
      setIsCheckboxSelected(newSelectedClusters.length > 0);
      return newSelectedClusters;
    });
  };

  const handlePreviousPage = () => {
    if (selectedIteration > 1) {
      setSelectedIteration(prev => prev - 1);
      setCurrentIterationUmapIndex(0);
    }
  };

  const handleNextPage = () => {
    if (selectedIteration < state.currentIteration) {
      setSelectedIteration(prev => prev + 1);
      setCurrentIterationUmapIndex(0);
    }
  };

  const handleSubmitClusters = async () => {
    try {
      const response = await axios.post('http://localhost:5000/submit-clusters', {
        selectedClusters: localSelectedClusters,
        iteration: selectedIteration
      });
      console.log('Clusters submitted successfully:', response.data);
    } catch (error) {
      setError('Failed to submit clusters. Please try again.');
    }
  };  

  const handleAutoFillIn = async () => {
    try {
      const response = await axios.post('http://localhost:5000/auto-fill-in');
      const { umap: newUmapFilename, newClusterInfo } = response.data;
      console.log("AutoFillIn triggered and new UMAP generated:", newUmapFilename);
      const newUmapUrl = `http://localhost:5000/umap/${newUmapFilename}`;
      addUmapData({ url: newUmapUrl, iteration: state.currentIteration, annotationDict: newClusterInfo, type: 'complete', order: 0 });
    } catch (error) {
      setError('Failed to trigger AutoFillIn. Please try again.');
    }
  };
  

  const handleZoomIn = async () => {
    if (localSelectedClusters.length > 0) {
      setIsZoomingIn(true);
      try {
        const response = await axios.post('http://localhost:5000/zoom-in', {
          selectedClusters: localSelectedClusters,
          resolution: resolution
        });
        const { umap: newUmapFilename, newClusterInfo } = response.data;
        const newUmapUrl = `http://localhost:5000/umap/${newUmapFilename}`;
        addUmapData({ url: newUmapUrl, iteration: state.currentIteration, annotationDict: newClusterInfo, type: 'zoomed', order: 0 });
        setSelectedIteration(state.umapData.length - 1);
        setIsZoomedIn(true);
        setLocalSelectedClusters([]);
        setClusterInfo(newClusterInfo);
      } catch (error) {
        setError('Failed to zoom in. Please try again.');
      } finally {
        setIsZoomingIn(false);
        setIsZoomedIn(true);
      }
    }
  };

  const handleZoomOut = async () => {
    if (isZoomedIn) {
      setIsZoomingOut(true);
      try {
        const response = await axios.post('http://localhost:5000/zoom-out');
        const { umap: newUmapFilename, newClusterInfo } = response.data;
        const newUmapUrl = `http://localhost:5000/umap/${newUmapFilename}`;
        addUmapData({ url: newUmapUrl, iteration: state.currentIteration, annotationDict: newClusterInfo, type: 'complete', order: 0 });
        setSelectedIteration(state.umapData.length - 1);
        setIsZoomedIn(false);
        console.log('[DEBUG] new cluster info from zoom out endpoint: ', newClusterInfo);
        setClusterInfo(newClusterInfo);
      } catch (error) {
        setError('Failed to zoom out. Please try again.');
      } finally {
        setIsZoomingOut(false);
        setIsZoomedIn(false);
      }
    }
  };

  const handleSwitch = useCallback(() => {
    if (currentIterationUmaps.length > 1) {
      setCurrentIterationUmapIndex(prev => (prev + 1) % currentIterationUmaps.length);
    }
  }, [currentIterationUmaps]);

  //====================
  // Render helpers
  //====================
  const renderClusterCheckboxes = () => {
    // if current iteration is not the last one, dont render the checkboxes
    if (!isLastUmap) {
      return null;
    }
    
    const currentResult = analysisResults[selectedIteration - 1];
    if (!currentResult) {
      return null;
    }
    
    let annotationDict = currentResult.annotation_dict || {};
    if (clusterInfo && Object.keys(clusterInfo).length > 0) {
      console.log('[DEBUG] using clusterInfo: ', clusterInfo);
      annotationDict = clusterInfo;
    }

    return (
      <div key={checkboxKey}>
        {Object.entries(annotationDict).map(([clusterId, clusterName]) => (
          <FormControlLabel
            key={clusterId}
            control={
              <Checkbox
                checked={localSelectedClusters.includes(clusterId)}
                onChange={() => handleClusterSelect(clusterId)}
              />
            }
            label={`${clusterName} (${clusterId})`}
          />
        ))}
      </div>
    );
  };

  const { currentStage } = usePipeline();

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        UMAP Plot
      </Typography>
      <Paper elevation={3} sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
        <IconButton
          onClick={handleSwitch}
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            zIndex: 1,
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
            },
          }}
        >
          <SwapHorizIcon />
        </IconButton>
        {umapData.length === 0 ? (
          <Typography>No UMAP plot is available</Typography>
        ) : currentUmapData ? (
          <>
            <Box sx={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              {isLoading && <CircularProgress />}
              {!isLoading && !error && (
                <img 
                  src={currentUmapData.url} 
                  alt={`UMAP Plot`} 
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                />
              )}
              {error && <Typography color="error">{error}</Typography>}
            </Box>
            {!isZoomedIn && (
              <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
                {renderClusterCheckboxes()}
              </Box>
            )}
            {(isZoomingIn || isZoomingOut) && <LinearProgress />}
            {!isZoomedIn && isLastUmap && currentStage === 'evaluation' && (
              <Box sx={{ mt: 2, mb: 2, width: '100%', maxWidth: '500px' }}>
                <Typography variant="body2" gutterBottom>Resolution:</Typography>
                <Slider
                  value={resolution}
                  onChange={(_, newValue) => setResolution(newValue as number)}
                  min={0.1}
                  max={2}
                  step={0.05}
                  valueLabelDisplay="auto"
                  disabled={!isCheckboxSelected}
                />
              </Box>
            )}
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
              <Button 
                onClick={handlePreviousPage} 
                disabled={!isPrevIterationAvailable}
              >
                Previous
              </Button>
              <Button 
                onClick={handleSubmitClusters}
                disabled={localSelectedClusters.length === 0 || isZoomingOut}
              >
                Stabilize Result
              </Button>
              <Button
                onClick={handleAutoFillIn}
                disabled={isZoomingOut}
              >
                Auto Fill In
              </Button>
              <Button 
                onClick={handleZoomIn} 
                disabled={!isCheckboxSelected || !isLastUmap || isZoomedIn || currentStage !== 'evaluation' || isZoomingIn || isZoomingOut}
              >
                Zoom In
              </Button>
              <Button 
                onClick={handleZoomOut}
                disabled={!isLastUmap || currentStage !== 'evaluation' || !isZoomedIn || isZoomingOut || isZoomingIn}
              >
                Zoom Out  
              </Button>
              <Button 
                onClick={handleNextPage} 
                disabled={!isNextIterationAvailable}
              >
                Next
              </Button>
            </Box>
          </>
        ) : (
          <Typography>Error: No UMAP plot is available for this iteration</Typography>
        )}
      </Paper>
      <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default UMAPPlot;