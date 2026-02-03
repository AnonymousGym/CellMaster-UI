import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, Paper, Button, CircularProgress } from '@mui/material';
import { useAppContext } from '../hooks/useAppContext';

const DotPlot: React.FC = () => {
  //====================
  // State management
  //====================
  const { state } = useAppContext();
  const [currentPage, setCurrentPage] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  //====================
  // Derived state
  //====================
  const { dotplotUrls } = state;
  const currentDotplotUrl = dotplotUrls[currentPage];

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
    if (dotplotUrls.length > 0) {
      setCurrentPage(dotplotUrls.length - 1);
    }
  }, [dotplotUrls]);

  useEffect(() => {
    if (currentDotplotUrl) {
      loadImage(currentDotplotUrl);
    }
  }, [currentDotplotUrl, loadImage]);

  //====================
  // Event handlers
  //====================
  const handlePrevious = () => {
    setCurrentPage((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentPage((prev) => Math.min(dotplotUrls.length - 1, prev + 1));
  };

  //====================
  // Render
  //====================
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        Gene Expression Dot Plot
      </Typography>
      <Paper elevation={3} sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        {dotplotUrls.length > 0 ? (
          <>
            <Box sx={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              {isLoading && <CircularProgress />}
              {error && <Typography color="error">{error}</Typography>}
              {!isLoading && !error && (
                <img 
                  key={currentDotplotUrl}
                  src={currentDotplotUrl} 
                  alt={`Gene Expression Dot Plot ${currentPage + 1}`} 
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                />
              )}
            </Box>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <Button onClick={handlePrevious} disabled={currentPage === 0 || dotplotUrls.length === 0}>
                Previous
              </Button>
              <Typography>
                {dotplotUrls.length > 0 ? `${currentPage + 1} / ${dotplotUrls.length}` : 'No dotplots'}
              </Typography>
              <Button onClick={handleNext} disabled={currentPage === dotplotUrls.length - 1 || dotplotUrls.length === 0}>
                Next
              </Button>
            </Box>
          </>
        ) : (
          <Typography variant="body1">No dotplot available</Typography>
        )}
      </Paper>
    </Box>
  );
};

export default DotPlot;