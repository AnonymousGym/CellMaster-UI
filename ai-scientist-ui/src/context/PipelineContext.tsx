import React, { createContext, useState, useContext, ReactNode } from 'react';
import { useAppContext } from '../hooks/useAppContext';

//====================
// Types
//====================
export type StageType = 'idle' | 'hypothesis' | 'experiment' | 'environment' | 'optionalRulesDuplet' | 'optionalRulesContamination' | 'evaluation';

interface PipelineContextType {
  currentStage: StageType;
  setCurrentStage: (stage: StageType) => void;
  isFeedbackEnabled: boolean;
  setIsFeedbackEnabled: (enabled: boolean) => void;
  feedbackPrompt: string;
  setFeedbackPrompt: (prompt: string) => void;
  isPipelineRunning: boolean;
  setIsPipelineRunning: (running: boolean) => void;
  handleFeedbackSubmit: (feedback: string) => void;
}

//====================
// Context
//====================
const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

//====================
// Provider Component
//====================
export const PipelineProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { setCurrentFeedback } = useAppContext();

  //====================
  // State
  //====================
  const [currentStage, setCurrentStage] = useState<StageType>('idle');
  const [isFeedbackEnabled, setIsFeedbackEnabled] = useState(false);
  const [feedbackPrompt, setFeedbackPrompt] = useState('');
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);

  //====================
  // Handlers
  //====================
  const handleFeedbackSubmit = (feedback: string) => {
    setIsFeedbackEnabled(false);
    setFeedbackPrompt('');
    setCurrentFeedback(feedback);
    
    console.log("handleFeedbackSubmit", currentStage);
    switch (currentStage) {
      case 'hypothesis':
        setCurrentStage('experiment');
        break;
      case 'experiment':
        setCurrentStage('environment');
        break;
      case 'environment':
        setCurrentStage('optionalRulesDuplet');
        break;
      case 'optionalRulesDuplet':
        setCurrentStage('optionalRulesContamination');
        break;
      case 'optionalRulesContamination':
        setCurrentStage('evaluation');
        break;
      case 'evaluation':
        setCurrentStage('hypothesis');
        break;
      default:
        setCurrentStage(prevStage => {
          const stages: StageType[] = ['hypothesis', 'experiment', 'environment', 'optionalRulesDuplet', 'optionalRulesContamination', 'evaluation'];
          const currentIndex = stages.indexOf(prevStage);
          return stages[(currentIndex + 1) % stages.length];
        });
    }
    console.log("handleFeedbackSubmit done", currentStage);
    setIsPipelineRunning(true);
  };

  //====================
  // Context Value
  //====================
  const contextValue: PipelineContextType = {
    currentStage,
    setCurrentStage,
    isFeedbackEnabled,
    setIsFeedbackEnabled,
    feedbackPrompt,
    setFeedbackPrompt,
    isPipelineRunning,
    setIsPipelineRunning,
    handleFeedbackSubmit,
  };

  return (
    <PipelineContext.Provider value={contextValue}>
      {children}
    </PipelineContext.Provider>
  );
};

//====================
// Custom Hook
//====================
export const usePipeline = () => {
  const context = useContext(PipelineContext);
  if (context === undefined) {
    throw new Error('usePipeline must be used within a PipelineProvider');
  }
  return context;
};