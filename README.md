
# CellMaster-UI

An AI-powered interactive web application for automated single-cell RNA-seq cell type annotation. CellMaster-UI combines hypothesis-driven analysis with advanced machine learning agents to provide accurate, iterative cell type identification.

## Features

- **Interactive Workflow**: Upload data, provide hypotheses, and refine annotations through an intuitive UI
- **Multiple Annotation Methods**: Compare results from CellTypist, GPTCellType, and the custom CellMaster pipeline
- **Visual Analytics**: Real-time UMAP plots, dot plots, and marker gene visualization
- **Iterative Refinement**: Human-in-the-loop feedback system for improving annotation accuracy

## Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher
- **npm**: 6.x or higher
- **R**: Required for Cell Ontology (CL) lookups
- **OpenAI API Key**: Required for AI-powered features

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CellMaster-UI
```

### 2. Set Up API Key
Edit `/config/settings.py` and replace the placeholder with your OpenAI API key:
```python
OPENAI_API_KEY = "your-api-key-here"
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
cd server
pip install -r requirements.txt
cd ..
```

### 4. Install Frontend Dependencies
```bash
cd ai-scientist-ui
npm install
cd ..
```

### 5. Set Up R Environment
The application uses R for ontology lookups. Ensure R is installed and the `rols` package is available:
```R
install.packages("rols")
```

## 🎯 Usage

### Starting the Application

#### 1. Start the Backend Server
Open a terminal and run:
```bash
cd server
python app.py
```
The server will start on `http://localhost:5000`

#### 2. Start the Frontend UI
Open a new terminal and run:
```bash
cd ai-scientist-ui
npm start
```
The UI will open automatically at `http://localhost:3000`

### Using the Interface

#### Step 1: Upload Your Data
1. **H5AD File** (Required): Upload your single-cell RNA-seq data in H5AD format
2. **Marker Genes CSV** (Optional): Upload a CSV file with cluster-specific marker genes
3. **Original Grouping Column**: Specify the column name in your H5AD file that contains cluster assignments (e.g., "leiden", "seurat_clusters")
4. **CellTypist Model** (Optional): Specify a CellTypist model name for comparison (e.g., "Healthy_Adult_Heart.pkl")

#### Step 2: Provide Initial Hypothesis
Enter your hypothesis about the tissue type or expected cell types. For example:
- "This is liver tissue"
- "PBMC sample containing immune cells"
- "Retinal tissue with photoreceptor cells"

#### Step 3: Run Analysis
Click **"Upload and Hypothesis"** to start the annotation pipeline. The system will:
1. Load and preprocess your data
2. Generate marker gene signatures
3. Query AI models for cell type predictions
4. Compare with CellTypist annotations
5. Display results with confidence scores

#### Step 4: Review Results
The interface displays:
- **Analysis Results Panel**: Iteration history, cluster annotations, and confidence metrics
- **Dot Plot**: Marker gene expression across clusters
- **UMAP Plot**: Spatial visualization of cell populations with annotations

#### Step 5: Explore and Refine (Optional)

- Zoom in and out to change granuality of clustering
- Request re-annotation of uncertain clusters
- Provide feedback to refine predictions

## Input File Formats

### H5AD File
Standard AnnData format with:
- `.X`: Expression matrix (genes × cells)
- `.obs`: Cell metadata including cluster assignments
- `.var`: Gene metadata

### Marker Genes CSV
CSV file with columns:
- `cluster`: Cluster identifier
- `gene`: Gene symbol
- `avg_log2FC` (or similar): Fold change metric
- Additional metrics as available

Example:
```csv
cluster,gene,p_val,avg_log2FC,pct.1,pct.2
0,CD3D,0.001,2.5,0.9,0.1
0,CD3E,0.002,2.3,0.85,0.15
1,CD79A,0.001,3.1,0.95,0.05
```

## Output Files

The application generates outputs in the following directories:

### `/outputs/` directory:
- `annotation_dict_*.txt`: Cluster-to-cell-type mappings for each iteration
- `*_umap_plot.png`: UMAP visualizations with annotations
- `dot_plot_*.png`: Marker gene dot plots

### `/uploads/` directory:
- Uploaded input files are stored here

## Benchmark Results (running evaluation):

run generate_score.py for scoring the annotation

### Adjusting Parameters
Edit variables at the top of `generate_score.py`:
```python
input_dir = "uploads/"
h5ad_file = "your_file.h5ad"
markers_file = 'your_markers.csv'
original_grouping = "leiden"
correct_column = "ground_truth"  # If available for benchmarking
threshold = 0.95  # Confidence threshold
tissue_name = "your_tissue"
```

### Cell Type Mapping
The `cell_type_mapping` dictionary in `generate_score.py` can be customized to standardize cell type names across different nomenclatures.

## Troubleshooting

### Common Issues

**R Environment Not Found**
Ensure R_HOME is set correctly. The Default Setting is: 
```bash
export R_HOME=/Library/Frameworks/R.framework/Resources
```

## Project Structure

```
CellMaster-UI/
├── ai-scientist-ui/        # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── context/        # React context providers
│   │   └── types/          # TypeScript definitions
├── server/                 # Flask backend
│   ├── app.py             # Main server application
│   ├── uploads/           # User uploaded files
│   └── outputs/           # Generated results
├── agents/                # AI agent modules
│   ├── hypothesis_agent/
│   ├── experiment_agent/
│   ├── evaluation_agent/
│   └── environment_agent/
├── config/                # Configuration files
├── utils/                 # Utility functions
└── generate_score.py  # Evaluation script
```
