import React, { useState, useEffect } from 'react';
import { Box, Typography, Tabs, Tab, Button } from '@mui/material';
import { useAppContext } from '../hooks/useAppContext';
import ReactMarkdown from 'react-markdown';

//====================
// Types
//====================
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

//====================
// Helper Components
//====================
const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
      sx={{ p: 2, maxHeight: 400, overflowY: 'auto' }}
    >
      {value === index && <Box>{children}</Box>}
    </Box>
  );
};

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

//====================
// Main Component
//====================
const AnalysisResults: React.FC = () => {
  //====================
  // Hooks
  //====================
  const { state, addDotplotUrl, addUmapData } = useAppContext();
  const { analysisResults } = state;

  //====================
  // State
  //====================
  const [tabValue, setTabValue] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);

  //====================
  // Event Handlers
  //====================
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handlePrevious = () => {
    setCurrentPage((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentPage((prev) => Math.min(analysisResults.length - 1, prev + 1));
  };

  //====================
  // Effects
  //====================
  useEffect(() => {
    if (analysisResults.length > 0) {
      setCurrentPage(analysisResults.length - 1);
    } else {
      setCurrentPage(0);
    }
  }, [analysisResults]);

  useEffect(() => {
    const currentResult = analysisResults[currentPage];
    if (currentResult?.dotplot) {
      const newDotplotUrl = `http://localhost:5000/dotplot/${currentResult.dotplot}`;
      if (!state.dotplotUrls.includes(newDotplotUrl)) {
        addDotplotUrl(newDotplotUrl);
      }
    }
    if (currentResult?.umap) {
      const newUmapUrl = `http://localhost:5000/umap/${currentResult.umap}`;
      if (!state.umapData.some(umap => umap.url === newUmapUrl)) {
        addUmapData({ url: newUmapUrl, iteration: state.currentIteration, annotationDict: currentResult.annotation_dict, type: 'complete', order: 0 });
      }
    }
  }, [currentPage, analysisResults, addDotplotUrl, addUmapData, state.dotplotUrls, state.umapData, state.currentIteration]);

  //====================
  // Helper Functions
  //====================
  const currentResult = analysisResults[currentPage];

  const getContent = (type: 'hypothesis' | 'experiment' | 'evaluation') => {
    if (!currentResult) return 'No results available yet.';
    return currentResult[type] || 'No results available for this stage.';
  };

  //====================
  // Render
  //====================
  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h6" gutterBottom>Analysis Results</Typography>
      <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis results tabs">
        <Tab label="Hypothesis" {...a11yProps(0)} />
        <Tab label="Experiment" {...a11yProps(1)} />
        <Tab label="Evaluation" {...a11yProps(2)} />
      </Tabs>
      <TabPanel value={tabValue} index={0}>
        <ReactMarkdown>{getContent('hypothesis')}</ReactMarkdown>
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        <ReactMarkdown>{getContent('experiment')}</ReactMarkdown>
      </TabPanel>
      <TabPanel value={tabValue} index={2}>
        <ReactMarkdown>{getContent('evaluation')}</ReactMarkdown>
      </TabPanel>
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', width: '100%' }}>
        <Button onClick={handlePrevious} disabled={currentPage === 0 || analysisResults.length === 0}>
          Previous
        </Button>
        <Typography>
          Iteration {analysisResults.length > 0 ? currentPage + 1 : 0} / {analysisResults.length}
        </Typography>
        <Button onClick={handleNext} disabled={currentPage === analysisResults.length - 1 || analysisResults.length === 0}>
          Next
        </Button>
      </Box>
    </Box>
  );
};

export default AnalysisResults;