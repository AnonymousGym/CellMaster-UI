import React, { useState } from 'react';
import { Box, TextField, Typography, Paper, Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import { usePipeline } from '../context/PipelineContext';
import { useAppContext } from '../hooks/useAppContext';

//====================
// Constants
//====================
const stageFeedbackPrompts = {
  hypothesis: "Review and adjust the refined hypothesis:",
  experiment: "Modify or approve the proposed experiment:",
  evaluation: "Comment on or guide the interpretation of results:",
  environement:"Provide observations about duplet cells or contamination (optional):",
  optionalRulesDuplet: "Provide any other feedback about annotation (optional):",
  optionalRulesContamination: "Provide any other feedback about annotation (optional):",
  newIteration: "Provide guidance for the next round or press Enter to continue:"
};

const optionalRulesExamples = {
  duplet: 
  `
  Duplet Example: Cluster 5: CD3D+CD19+ (This suggests cluster 5 might contain T cells (CD3D) and B cells (CD19) together) 
  Contamination Example: LYZ2: unexpectedly high in T cell clusters (This suggests the T cell marker LYZ2 is expressed in unexpected clusters) 
  `,
  contamination: "Any particular gene or cluster of interest: have you decided on a single cluster or gene?"
};

//====================
// Styled Components
//====================
const HighlightedPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  backgroundColor: theme.palette.info.light,
  color: theme.palette.info.contrastText,
}));

const ExploreInput: React.FC = () => {
  //====================
  // State management
  //====================
  const [exploreQuery, setExploreQuery] = useState('');
  const { isFeedbackEnabled, currentStage, feedbackPrompt, handleFeedbackSubmit } = usePipeline()
  const { state } = useAppContext();

  //====================
  // Derived state
  //====================
  const isDotplotAvailable = state.dotplotUrls.length > 0;
  const isInputDisabled = !isFeedbackEnabled || (currentStage.startsWith('optionalRules') && !isDotplotAvailable);

  //====================
  // Event handlers
  //====================
  const handleSubmit = () => {
    console.log("handleSubmit", currentStage);
    if (isFeedbackEnabled) {
      switch (currentStage) {
        case 'hypothesis':
          handleFeedbackSubmit(exploreQuery);
          break;
        case 'experiment':
          handleFeedbackSubmit(exploreQuery);
          break;
        case 'environment':
          handleFeedbackSubmit(exploreQuery);
          break;
        case 'optionalRulesDuplet':
          handleFeedbackSubmit(exploreQuery);
          break;
        case 'optionalRulesContamination':
          handleFeedbackSubmit(exploreQuery);
          break;
        case 'evaluation':
          handleFeedbackSubmit(exploreQuery);
          break;
        default:
      }
      setExploreQuery('');
    }
    console.log("handleSubmit done", currentStage);
  };

  //====================
  // Render helpers
  //====================
  const getPlaceholder = () => {
    if (!isFeedbackEnabled) return "Waiting for feedback to be enabled...";
    if (currentStage === 'environment') return optionalRulesExamples.duplet;
    if (currentStage === 'optionalRulesDuplet') return optionalRulesExamples.contamination;
    return stageFeedbackPrompts[currentStage as keyof typeof stageFeedbackPrompts] || "Enter your feedback";
  };

  //====================
  // Render
  //====================
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {stageFeedbackPrompts[currentStage as keyof typeof stageFeedbackPrompts] || "Explore Next"}
      </Typography>
      {isFeedbackEnabled && feedbackPrompt && (
        <HighlightedPaper elevation={3}>
          <Box sx={{ whiteSpace: 'pre-wrap' }}>
            <Typography variant="body1">{feedbackPrompt}</Typography>
          </Box>
        </HighlightedPaper>
      )}
      <TextField
        fullWidth
        variant="outlined"
        value={exploreQuery}
        onChange={(e) => setExploreQuery(e.target.value)}
        placeholder={getPlaceholder()}
        multiline
        rows={4}
        disabled={isInputDisabled}
        sx={{
          '& .MuiInputBase-input.Mui-disabled': {
            WebkitTextFillColor: '#000000',
          },
          marginBottom: 2,
        }}
      />
      <Button
        variant="contained"
        color="primary"
        onClick={handleSubmit}
        disabled={isInputDisabled}
        fullWidth
      >
        Submit Feedback
      </Button>
    </Box>
  );
};

export default ExploreInput;