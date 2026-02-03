import React from 'react';
import { Box, Container, Grid, Paper, ThemeProvider, CssBaseline } from '@mui/material';
import { createTheme } from '@mui/material/styles';
import Header from './components/Header';
import HypothesisInput from './components/HypothesisInput';
import AnalysisResults from './components/AnalysisResults';
import DotPlot from './components/DotPlot';
import UMAPPlot from './components/UMAPPlot';
import ExploreInput from './components/ExploreInput';
import { AppProvider } from './context/AppContext';
import { PipelineProvider } from './context/PipelineContext';
import PipelineManager from './components/PipelineManager';
import { usePipeline } from './context/PipelineContext';

const theme = createTheme({
  palette: {
    background: {
      default: '#f5f5f5',
    },
  },
});

function AppContent() {
  const { isFeedbackEnabled } = usePipeline();
  const MemoizedExploreInput = React.memo(ExploreInput);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header />
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        <PipelineManager />
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
              <HypothesisInput />
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
              <AnalysisResults />
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
              <DotPlot />
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
              <UMAPPlot />
            </Paper>
          </Grid>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <MemoizedExploreInput />
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

function App() {
  return (
    <AppProvider>
      <PipelineProvider>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AppContent />
        </ThemeProvider>
      </PipelineProvider>
    </AppProvider>
  );
}

export default App;