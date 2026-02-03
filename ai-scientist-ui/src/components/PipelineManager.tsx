import React, { useEffect, useCallback } from 'react';
import axios from 'axios';
import { useAppContext } from '../hooks/useAppContext';
import { usePipeline, StageType } from '../context/PipelineContext';
import { AnalysisResult } from '../context/AppContext';

//====================
// Types
//====================
type PayloadType = {
  stage: StageType;
  hypothesis: string;
  original_grouping : string;
  iteration: number;
  annotation_dict: any | null;
  no_gene_cluster: any | null;
  iteration_summary: any | null;
  optionalRules?: {
    duplet: string;
    contamination: string;
  };
  feedback?: any | null;
};

//====================
// Main Component
//====================
const PipelineManager: React.FC = () => {
  //====================
  // Hooks
  //====================
  const { state, addAnalysisResult, updateLatestAnalysisResult, addDotplotUrl, addUmapData, setIsPipelineRunning, incrementIteration } = useAppContext();
  const { 
    currentStage, 
    setCurrentStage, 
    setFeedbackPrompt, 
    isFeedbackEnabled, 
    setIsFeedbackEnabled,
  } = usePipeline();
  const API_URL = 'http://localhost:5000';

  //====================
  // Pipeline Stage Runner
  //====================
  const runPipelineStage = useCallback(async () => {
    try {
      const payload: PayloadType = {
        stage: currentStage,
        hypothesis: state.hypothesis,
        original_grouping:state.original_grouping,
        iteration: state.currentIteration,
        annotation_dict: state.analysisResults[state.analysisResults.length - 1]?.annotation_dict || null,
        no_gene_cluster: state.analysisResults[state.analysisResults.length - 1]?.no_gene_cluster || null,
        iteration_summary: state.analysisResults[state.analysisResults.length - 1]?.iteration_summary || null,
        feedback: state.currentFeedback || null,
      };

      if (currentStage === 'evaluation') {
        payload.optionalRules = state.optionalRules;
      }

      const response = await axios.post(`${API_URL}/process`, payload, {
        withCredentials: true,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      const newResult: Partial<AnalysisResult> = {
        [currentStage]: response.data.result,
      };
    
      switch (currentStage) {
        case 'hypothesis':
          incrementIteration();
          setIsFeedbackEnabled(true);
          setFeedbackPrompt("After reading the hypothesis, do you find anything important left out? Do you want specific subtypes for certain cell types (like CD4 T and CD8 T)? Should we include time / group wise annotations, such as XX-young and XX-old?");
          addAnalysisResult(newResult as AnalysisResult);
          break;
        case 'experiment':
          if (response.data.adjusted_hypothesis) {
            updateLatestAnalysisResult({ hypothesis: response.data.adjusted_hypothesis });
          }
          updateLatestAnalysisResult(newResult);
          setIsFeedbackEnabled(true);
          setFeedbackPrompt("Now we have a marker gene list proposed, which will be used to generate a dot plot. Please refer to the marker gene list and related potential cell type descriptions. Provide any modifications or add-ons you like here.");
          break;
        case 'environment':
          if (response.data.adjusted_experiment) {
            updateLatestAnalysisResult({ experiment: response.data.adjusted_experiment });
          }
          if (response.data.dotplot) {
            newResult.dotplot = response.data.dotplot;
            const newDotplotUrl = `${API_URL}/dotplot/${response.data.dotplot}`;
            addDotplotUrl(newDotplotUrl);
          }
          updateLatestAnalysisResult(newResult);
          setIsFeedbackEnabled(true);
          setFeedbackPrompt("Provide observations about duplet cells or contamination (optional): duplet means one cluster contains two or more cell types. contamination means one gene is overly expressed in too many clusters. Tell us about these cluster numbers and genes.");
          break;
        case 'optionalRulesDuplet':
          setIsFeedbackEnabled(true);
          setFeedbackPrompt("Do you have other important insights for our annotation? This could be any single cluster annotation or usage of any gene: E.g. 'use Nkg7 for NK cell' ");
          break;
        case 'optionalRulesContamination':
          setCurrentStage('evaluation');
          break;
        case 'evaluation':
          newResult.annotation_dict = response.data.result.annotation_dict;
          newResult.no_gene_cluster = response.data.result.no_gene_cluster;
          newResult.iteration_summary = response.data.result.iteration_summary;
          newResult.evaluation = response.data.result.evaluation;
          newResult.umap = response.data.umap;
          newResult.h5ad = response.data.h5ad;
          if (response.data.umap) {
            const newUmapUrl = `http://localhost:5000/umap/${response.data.umap}`;
            const type = response.data.umap.includes('zoomed') ? 'zoomed' : 'complete';
            addUmapData({ url: newUmapUrl, iteration: state.currentIteration, annotationDict: newResult.annotation_dict, type: type, order: 0 });
          }
          setIsFeedbackEnabled(true);
          setFeedbackPrompt("Review the evaluation results and provide feedback or press Enter to start a new iteration.");
          updateLatestAnalysisResult(newResult);
          break;
      }

    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error('[PipelineManager] Axios error details:', error.response?.data);
        console.error(`[PipelineManager] An error occurred during the ${currentStage} stage.`);
      }
      setFeedbackPrompt(` The pipeline has been stopped. Please review and restart if needed.`);
      setCurrentStage('idle');
      setIsFeedbackEnabled(false);
      setIsPipelineRunning(false);
    }
  }, [currentStage, state, API_URL, addAnalysisResult, updateLatestAnalysisResult, setCurrentStage, setIsFeedbackEnabled, setFeedbackPrompt, addDotplotUrl, addUmapData, setIsPipelineRunning, incrementIteration]);

  //====================
  // Effects
  //====================
  useEffect(() => {
    if (currentStage !== 'idle' && !isFeedbackEnabled && state.isPipelineRunning) {
      runPipelineStage();
    }
  }, [currentStage, isFeedbackEnabled, state.isPipelineRunning, runPipelineStage]);

  return null;
};

export default PipelineManager;