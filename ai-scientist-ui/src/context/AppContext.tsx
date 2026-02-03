import React, { createContext, useState, ReactNode } from 'react';

//====================
// Types
//====================
export type AnalysisResult = {
  hypothesis?: string;
  experiment?: string;
  evaluation?: any;
  dotplot?: string;
  umap?: string;
  h5ad?: string;
  subset_h5ad?: string | null;
  failedGenes?: string[];
  successfulGenes?: string[];
  annotation_dict?: any;
  no_gene_cluster?: any;
  iteration_summary?: any;
  optionalRules?: {
    duplet: string;
    contamination: string;
  };
};

export type UMAPData = {
  url: string;
  iteration: number;
  annotationDict: Record<string, string>;
  type: 'complete' | 'zoomed';
  order: number;
};

export type AppState = {
  hypothesis: string;
  original_grouping:string;
  h5adFile: FileWithPath | null;
  metadataFile: FileWithPath | null;
  currentIteration: number;
  analysisResults: AnalysisResult[];
  dotplotUrls: string[];
  optionalRules: {
    duplet: string;
    contamination: string;
  };
  isPipelineRunning: boolean;
  selectedClusters: string[];
  umapData: UMAPData[];
  currentUMAPIndex: number;
  currentFeedback: string | null;
};

type AppContextType = {
  state: AppState;
  setHypothesis: (hypothesis: string) => void;
  setOriginalGrouping:(original_grouping:string) => void;
  setH5adFile: (file: File | null) => void;
  setMetadataFile: (file: File | null) => void;
  addAnalysisResult: (result: AnalysisResult) => void;
  updateLatestAnalysisResult: (result: Partial<AnalysisResult>) => void;
  incrementIteration: () => void;
  addDotplotUrl: (url: string) => void;
  setOptionalRules: (rules: { duplet: string; contamination: string }) => void;
  setIsPipelineRunning: (isRunning: boolean) => void;
  setSelectedClusters: (clusters: string[]) => void;
  addUmapData: (newData: UMAPData) => void;
  updateLatestUmapData: (updatedData: Partial<UMAPData>) => void;
  setCurrentFeedback: (feedback: string | null) => void;
  currentUMAPIndex: number;
  resetApplication: () => void;
};

interface FileWithPath extends File {
  path?: string;
}

//====================
// Initial State
//====================
const initialState: AppState = {
  hypothesis: '',
  original_grouping:'',
  h5adFile: null,
  metadataFile: null,
  currentIteration: 0,
  analysisResults: [],
  dotplotUrls: [],
  optionalRules: {
    duplet: '',
    contamination: ''
  },
  isPipelineRunning: false,
  selectedClusters: [],
  umapData: [],
  currentUMAPIndex: 0,
  currentFeedback: null,
};

//====================
// Context
//====================
export const AppContext = createContext<AppContextType | undefined>(undefined);

//====================
// Provider Component
//====================
export const AppProvider: React.FC<{children: ReactNode}> = ({ children }) => {
  const [state, setState] = useState<AppState>(initialState);

  //====================
  // State Setters
  //====================
  const setHypothesis = (hypothesis: string) => setState(prev => ({ ...prev, hypothesis }));
  const setOriginalGrouping = (original_grouping: string) => setState(prev => ({ ...prev, original_grouping }));
  const setH5adFile = (h5adFile: File | null) => setState(prev => ({ ...prev, h5adFile }));
  const setMetadataFile = (metadataFile: File | null) => setState(prev => ({ ...prev, metadataFile }));
  const setOptionalRules = (optionalRules: { duplet: string; contamination: string }) => setState(prev => ({ ...prev, optionalRules }));
  const setIsPipelineRunning = (isPipelineRunning: boolean) => setState(prev => ({ ...prev, isPipelineRunning }));
  const setSelectedClusters = (selectedClusters: string[]) => setState(prev => ({ ...prev, selectedClusters }));
  const setCurrentFeedback = (currentFeedback: string | null) => setState(prev => ({ ...prev, currentFeedback }));

  //====================
  // State Updaters
  //====================
  const addAnalysisResult = (result: AnalysisResult) => {
    setState(prev => ({
      ...prev,
      analysisResults: [...prev.analysisResults, result]
    }));
  };

  const addUmapData = (newUMAPData: UMAPData) => {
    console.log('Existing UMAP data:', state.umapData);
    console.log('Adding UMAP data:', newUMAPData);
    setState(prev => {
      const updatedUMAPData = [...prev.umapData];
      const existingUMAPIndex = updatedUMAPData.findIndex(
        umap => umap.url === newUMAPData.url && umap.iteration === newUMAPData.iteration
      );
      if (existingUMAPIndex === -1) {
        const currentIterationUMAPs = updatedUMAPData.filter(
          umap => umap.iteration === newUMAPData.iteration
        );
        newUMAPData.order = currentIterationUMAPs.length;
        updatedUMAPData.push(newUMAPData);
      }
      return {
        ...prev,
        umapData: updatedUMAPData,
        currentUMAPIndex: updatedUMAPData.length - 1,
      };
    });
  };

  const updateLatestUmapData = (updatedData: Partial<UMAPData>) => {
    console.log('Existing UMAP data:', state.umapData);
    console.log('Updating UMAP data:', updatedData);
    setState(prev => {
      const newUmapData = [...prev.umapData];
      if (newUmapData.length > 0) {
        const lastIndex = newUmapData.length - 1;
        newUmapData[lastIndex] = { ...newUmapData[lastIndex], ...updatedData };
      }
      return { ...prev, umapData: newUmapData };
    });
  };
  
  const updateLatestAnalysisResult = (result: Partial<AnalysisResult>) => {
    setState(prev => {
      const updatedResults = [...prev.analysisResults];
      const lastIndex = updatedResults.length - 1;
      if (lastIndex >= 0) {
        updatedResults[lastIndex] = { ...updatedResults[lastIndex], ...result };
      }
      return { ...prev, analysisResults: updatedResults };
    });
  };
  
  const incrementIteration = () => setState(prev => ({ ...prev, currentIteration: prev.currentIteration + 1 }));
  
  const addDotplotUrl = (url: string) => {
    setState(prev => {
      if (!prev.dotplotUrls.includes(url)) {
        return { ...prev, dotplotUrls: [...prev.dotplotUrls, url] };
      }
      return prev;
    });
  };

  const resetApplication = () => {
    setState(initialState);
    // Also reset backend state
    fetch('http://localhost:5000/reset', { method: 'POST' })
      .then(() => {
        // Refresh the page after successful reset
        window.location.reload();
      })
      .catch(error => {
        console.error('Failed to reset backend state:', error);
        // Refresh anyway even if backend reset fails
        window.location.reload();
      });
  }

  //====================
  // Context Value
  //====================
  const contextValue: AppContextType = {
    state, 
    setHypothesis, 
    setOriginalGrouping,
    setH5adFile, 
    setMetadataFile, 
    addAnalysisResult,
    updateLatestAnalysisResult,
    incrementIteration,
    addDotplotUrl,
    setOptionalRules,
    setIsPipelineRunning,
    setSelectedClusters,
    addUmapData,
    updateLatestUmapData,
    setCurrentFeedback,
    currentUMAPIndex: state.currentUMAPIndex,
    resetApplication,
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};