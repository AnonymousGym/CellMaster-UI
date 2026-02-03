#====================
# Configuration
#====================
import sys
import traceback
import logging
import numpy as np
import seaborn as sns
import pandas as pd
import anndata as ad
import openai
import os
import warnings
from typing import Optional
warnings.filterwarnings("ignore")
# set up the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
ALLOWED_EXTENSIONS = {'h5ad', 'csv'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

#====================
# Imports
#====================
import numpy as np
import pandas as pd

from agents.hypothesis_agent.liver_hypothesis_agent import LiverHypothesisAgent
from agents.experiment_agent.liver_experiment_agent import LiverExperimentAgent
from agents.environment_agent.liver_environment_agent import LiverEnvironmentAgent
from agents.evaluation_agent.liver_evaluation_agent import LiverEvaluationAgent
from config.settings import OPENAI_API_KEY

from utils.liver_process_toolkit import solve_auto_fill_in

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import scanpy as sc
import matplotlib.pyplot as plt

import celltypist
from celltypist import models
import logging
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText

#====================
# App Setup
#====================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#====================
# Global Variables
#====================
hypothesis_agent = None
experiment_agent = None
environment_agent = None
evaluation_agent = None
current_iteration = 0
no_gene_cluster = []
failed_genes = []
successful_genes = []
annotation_dict = None
annotation_dict_list = []
analysis_history = []
optional_rules_contamination = None
optional_rules_duplet = None
adata = None
stabilized_annotation_dict = {}
initial_hypothesis = ""
original_grouping = ''
celltypist_model=""
app.config['H5AD_FILENAME'] = None
app.config['CSV_FILENAME'] = None  

dotplot = None

class FileConfig:
  def __init__(self):
    self.generate_csv_from_h5adh5ad_filename = None
    self.csv_filename = None 
    self.original_h5ad_filename: Optional[str] = None
    self.original_csv_filename: Optional[str] = None
    
file_config = FileConfig()

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_umap_other(adata, df_result):
    # Ensure UMAP is calculated
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.rank_genes_groups(adata, original_grouping, method='wilcoxon')
    # Convert original_grouping to string for mapping
    adata.obs[original_grouping] = adata.obs[original_grouping].astype(str)
    df_result[original_grouping] = df_result[original_grouping].astype(str)
    # Create mappings from clusters to labels
    cluster_to_predicted_labels = dict(zip(df_result[original_grouping], df_result['celltypist_predicted_labels']))
    cluster_to_majority_voting = dict(zip(df_result[original_grouping], df_result['celltypist_majority_voting']))
    cluster_to_gpt_predict = dict(zip(df_result[original_grouping], df_result['gptcelltype_predict']))
    # Add the mapped labels to the AnnData object
    adata.obs['celltypist_predicted_labels'] = adata.obs[original_grouping].map(cluster_to_predicted_labels)
    adata.obs['celltypist_majority_voting'] = adata.obs[original_grouping].map(cluster_to_majority_voting)
    adata.obs['gptcelltype_predict'] = adata.obs[original_grouping].map(cluster_to_gpt_predict)
    # Define columns to plot
    columns_to_plot = ['celltypist_predicted_labels', 'celltypist_majority_voting', 'gptcelltype_predict']
    # Plot UMAP for each column with original_grouping annotated and labels listed in a legend
    for col in columns_to_plot:
        plt.figure(figsize=(15, 15))  # Match the figure size
        
        # Use Scanpy's built-in UMAP plotting
        sc.pl.umap(adata, color=col, legend_loc=None, title=f'UMAP plot - {col}', show=False)
        
        # Add cluster numbers on top of UMAP points
        for cluster in sorted(adata.obs[original_grouping].unique(), key=lambda x: int(x)):
            # Calculate the center of each cluster
            mask = adata.obs[original_grouping] == cluster
            x_coords = adata.obsm['X_umap'][mask, 0]
            y_coords = adata.obsm['X_umap'][mask, 1]
            plt.text(x_coords.mean(), y_coords.mean(), cluster, fontsize=12, weight='bold', color='black')
        # Create a sorted list of clusters and corresponding legend text
        unique_clusters = sorted(adata.obs[original_grouping].unique(), key=lambda x: int(x))
        if col=="celltypist_predicted_labels":
            legend_text = "\n".join([f"Cluster {cluster}: {cluster_to_predicted_labels.get(cluster, 'N/A')}" for cluster in unique_clusters])
        elif col=="celltypist_majority_voting":
            legend_text = "\n".join([f"Cluster {cluster}: {cluster_to_majority_voting.get(cluster, 'N/A')}" for cluster in unique_clusters])
        elif col=="gptcelltype_predict":
            legend_text = "\n".join([f"Cluster {cluster}: {cluster_to_gpt_predict.get(cluster, 'N/A')}" for cluster in unique_clusters])
        # Position the legend outside the plot to prevent blocking the UMAP
        anchored_text = AnchoredText(legend_text, loc='center left', frameon=True, prop=dict(size=10), bbox_to_anchor=(1.05, 0.5), bbox_transform=plt.gca().transAxes)
        plt.gca().add_artist(anchored_text)
        # Save the plot with the original filenames
        umap_filename = f"{col}_umap_plot.png"
        plt.savefig(os.path.join(OUTPUT_FOLDER, umap_filename), dpi=300, bbox_inches='tight')
        plt.close()
    # Return the list of saved UMAP plot filenames
    return [os.path.join(OUTPUT_FOLDER, f"{col}_umap_plot.png") for col in columns_to_plot]

def generate_markers_csv(input_dir, output_dir, h5ad_file, groupby=original_grouping):
    """
    Generate markers.csv file from h5ad file using rank_genes_groups
    
    Args:
        input_dir (str): Directory containing h5ad file
        output_dir (str): Directory to save markers.csv
        h5ad_file (str): Name of h5ad file
        groupby (str): Column name to group cells by
    
    Returns:
        str: Name of generated markers.csv file
    """
    try:
        print(f'[DEBUG] Generating markers for {h5ad_file} with groupby={groupby}')
        # Read the AnnData object
        adata = sc.read_h5ad(os.path.join(input_dir, h5ad_file))
        
        # Compute rank_genes_groups
        sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon')
        
        # Convert to dataframe
        marker_df = sc.get.rank_genes_groups_df(adata, group=None)
        
        # Rename columns to match expected format
        marker_df = marker_df.rename(columns={
            'group': 'cluster',
            'names': 'gene',
            'logfoldchanges': 'avg_log2FC',
            'scores': 'score',
            'pvals_adj': 'p_val_adj'
        })
        
        # Sort by cluster and fold change
        marker_df = marker_df.sort_values(['cluster', 'avg_log2FC'], ascending=[True, False])
        
        # Save to CSV using standard name
        output_file = 'markers.csv'
        marker_df.to_csv(os.path.join(output_dir, output_file), index=False)
        print(f'[DEBUG] Generated markers file: {output_file}')
        
        return output_file
        
    except Exception as e:
        print(f'[ERROR] Failed to generate markers.csv: {str(e)}')
        return None

def generate_umap(iteration, is_subset, annotation_dict=None, groupby=original_grouping):
    global adata
    print('[DEBUG] Generating UMAP for iteration {}: is_subset={}, groupby={}'.format(iteration, is_subset, groupby))
    if not "umap" in adata.uns:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    plt.figure(figsize=(10, 10))
    
    complete_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith(f"{iteration}_") and 'complete' in f]
    zoomed_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith(f"{iteration}_") and 'zoomed' in f]
    umap_filename = None
    
    if is_subset:
      org_dict = {i: str(i) for i in adata.obs[groupby].astype(int).unique()}
      org_dict.update(annotation_dict)
      adata.obs['cellmaster_grouping_labels'] = adata.obs[groupby].astype(int).map(org_dict).astype('category')
      sc.pl.umap(adata, color='cellmaster_grouping_labels', legend_loc='on data', title='UMAP plot', show=False)
      umap_filename = f"{iteration}_{len(zoomed_files)}_{'zoomed'}_umap_plot.png"
    else:
      org_dict = {i: str(i) for i in adata.obs[groupby]}
      org_dict.update(annotation_dict)
      adata.obs[f"{original_grouping}_labels"] = adata.obs[groupby].astype(int).map(org_dict).astype('category')
      sc.pl.umap(adata, color=f"{original_grouping}_labels", legend_loc='on data', title='UMAP plot', show=False)
      umap_filename = f"{iteration}_{len(complete_files)}_{'complete'}_umap_plot.png"
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, umap_filename), dpi=300, bbox_inches='tight')
    plt.close()
    print('[DEBUG] UMAP plot saved as {}'.format(umap_filename))
    return umap_filename

def get_current_adata_path(default_h5ad_file=None):
    subset_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".h5ad")]
    latest_subset = None
    if subset_files:
        latest_subset = max(subset_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if latest_subset:
        print('[DEBUG] Latest subset found: {}'.format(latest_subset))
        return OUTPUT_FOLDER, latest_subset
    else:
        h5ad_filename = file_config.h5ad_filename or default_h5ad_file
        print('[DEBUG] No subset found, using original file: {}'.format(h5ad_filename))
        return UPLOAD_FOLDER, h5ad_filename

def merge_subset_to_main(data, subset_data, original_type, detail_type, merged_type):
    print(f"[DEBUG] Checking columns - Original type: {original_type}")
    print(f"[DEBUG] Available columns in main data: {list(data.obs.columns)}")
    print(f"[DEBUG] Available columns in subset data: {list(subset_data.obs.columns)}")
    data.obs[merged_type] = np.nan
    data.obs.loc[subset_data.obs.index, merged_type] = subset_data.obs[detail_type]
    col1_str = data.obs[merged_type].astype(str)
    col2_str = data.obs[original_type].astype(str)
    col1_str = col1_str.replace('nan', np.nan)
    merged_column = col1_str.fillna(col2_str)
    data.obs[merged_type] = pd.Categorical(merged_column)
    data.obs[original_type] = data.obs[merged_type]
    return data

#====================
# Routes
#====================
@app.route('/other_methods', methods=['GET'])
def other_methods():
    h5ad_file = file_config.h5ad_filename
    markers_file = file_config.csv_filename
    logging.info("Start celltypist function")
    adata = ad.read_h5ad(os.path.join(app.config['UPLOAD_FOLDER'], h5ad_file ))
    # adata = ad.read_h5ad("D:\AISCIENTIST\AIScientist\data\liver\input\liver.h5ad")
    df_result=get_celltypist(adata)
    
    logging.info("Start gptcelltype function")
    markers=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], markers_file ))
    # markers=pd.read_csv("D:\AISCIENTIST\AIScientist\data\liver\input\markers.csv")
    annotations=[]
    while (len(annotations)!=len(markers["cluster"].unique())):
        annotations,input_gene = gptcelltype(markers,tissuename="liver")

    df_result['gptcelltype_predict']=annotations
    df_result_dict = df_result.to_dict(orient='records')

    logging.info("Generating Umap")
    generate_umap_other(adata, df_result)

    # Prepare absolute paths to the images
    image_paths = {
        'celltypist_predicted_labels': 'celltypist_predicted_labels_umap_plot.png',
        'celltypist_majority_voting': 'celltypist_majority_voting_umap_plot.png',
        'gptcelltype_predict': 'gptcelltype_predict_umap_plot.png',
    }

    logging.info("Process Finished, All Results Attached")
    return jsonify({
        'status': 'UMAP plots generated and saved.',
        'image_paths': image_paths
    })

def gptcelltype(input_data,
                tissuename=None, 
                model='gpt-4', 
                topgenenumber=10, 
                batch_size=30,
                marker_filter_column="avg_log2FC",
                marker_cluster_column="cluster",
                gene_column="gene"):
    input_gene=[]
    if isinstance(input_data, list):
        input_data = [",".join(markers) for markers in input_data]
    elif isinstance(input_data, pd.DataFrame):
        # Process the Seurat FindAllMarkers DataFrame(not tested yet)
        if marker_filter_column in input_data.columns:
            print("filter_column found: ",marker_filter_column)
            input_data = input_data[input_data[marker_filter_column] > 0]
            input_data = input_data.sort_values(marker_filter_column, ascending=False) 
        if marker_cluster_column not in input_data.columns:
            print("cluster column not found: ",marker_cluster_column)
        input_data = input_data.groupby(marker_cluster_column)[gene_column].apply(lambda x: ','.join(x[:topgenenumber])).to_dict()
    if not OPENAI_API_KEY:
        print("API key not provided, returning the prompt.")
        return f"Identify cell types of {tissuename} cells using the following markers separately for each row.\n"
    else:
        try:
            # Batch processing
            results = []
            for i in range(0, len(input_data), batch_size):
                batch_input = input_data[i:i+batch_size] if isinstance(input_data, list) else list(input_data.values())[i:i+batch_size]
                input_gene.append(input_gene)
                batch_message_content = "\n".join(batch_input)
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"Identify cell types of {tissuename} cells using the following markers separately for each row.\nOnly provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n{batch_message_content}"}
                    ],
                    max_tokens=500,
                    temperature =0.3,
                )
                
                batch_results = response['choices'][0]['message']['content'].strip().split('\n')
                results.extend(batch_results)
            print("GPT CALL finished")
            if isinstance(input_data, list):
                return results,input_gene
            else:
                return results,input_gene
        except Exception as e:
            print(f"Error: {str(e)}")
            if isinstance(input_data, list):
                return [str(e)] * len(input_data)
            else:
                return {k: str(e) for k, v in input_data.items()}

def get_celltypist(adata):
    if original_grouping in adata.obs.columns:
        logging.info(f"Found {original_grouping} Clusters")
        sc.tl.rank_genes_groups(adata, original_grouping, method='wilcoxon')
    else:
        try:
            logging.info("Found cellmaster_grouping Clusters")
            sc.tl.rank_genes_groups(adata, 'cellmaster_grouping', method='wilcoxon')
        except:
            logging.info("Start generating cellmaster_grouping Clusters")
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            # copy the leiden column to cellmaster_grouping
            adata.obs['cellmaster_grouping'] = adata.obs['leiden']
            sc.tl.rank_genes_groups(adata, 'cellmaster_grouping', method='wilcoxon')

    logging.info("Start celltypist annotation")
    models.download_models(force_update=False)
    print('[DEBUG] Use Celltypist model {}'.format(celltypist_model))
    predictions = celltypist.annotate(adata, model=celltypist_model, majority_voting=True)
    adata_new = predictions.to_adata()

    if original_grouping not in adata_new.obs:
        raise ValueError(f"adata_new does not contain {original_grouping} in .obs")
    
    predicted_labels = adata_new.obs['predicted_labels']
    majority_votings = adata_new.obs['majority_voting']
    clusters = adata_new.obs[original_grouping]

    df = pd.DataFrame({
         original_grouping: clusters,
        'celltypist_predicted_labels': predicted_labels,
        'celltypist_majority_voting': majority_votings
    })

    def get_top_95_percent(group):
        count = group.value_counts(normalize=True)
        cumulative_sum = 0
        top_labels = []

        for label, proportion in count.items():
            cumulative_sum += proportion
            top_labels.append(label)
            if cumulative_sum >= 0.95:
                break

        return ','.join(top_labels)

    df_merged_grouped = df.groupby(original_grouping)['celltypist_predicted_labels'].apply(get_top_95_percent).reset_index()
    df_majority_grouped = df.groupby(original_grouping)['celltypist_majority_voting'].apply(get_top_95_percent).reset_index()
    df_grouped = pd.merge(df_merged_grouped, df_majority_grouped, on=original_grouping, how='inner', suffixes=('_merged', '_majority'))
    logging.info(f"Celltypist function ended, return:\n{df_grouped}")
    return df_grouped

@app.route('/upload', methods=['POST'])
def upload_files():
    global celltypist_model, original_grouping
    
    # Get the original grouping and celltypist model
    original_grouping = request.form.get('original_grouping')
    celltypist_model = request.form.get('celltypistModel', "Healthy_Mouse_Liver.pkl")
    
    # Handle h5ad file
    if 'h5ad' in request.files:
        h5ad_file = request.files['h5ad']
        file_config.original_h5ad_filename = h5ad_file.filename
        file_config.h5ad_filename = secure_filename(h5ad_file.filename)
        h5ad_path = os.path.join(app.config['UPLOAD_FOLDER'], file_config.h5ad_filename)
        h5ad_file.save(h5ad_path)
    elif 'h5ad_path' in request.form:
        h5ad_path = request.form['h5ad_path']
        file_config.original_h5ad_filename = os.path.basename(h5ad_path)
        file_config.h5ad_filename = secure_filename(os.path.basename(h5ad_path))
        try:
            # Verify file exists and is readable
            with open(h5ad_path, 'rb') as f:
                # Copy file to upload folder
                target_path = os.path.join(app.config['UPLOAD_FOLDER'], file_config.h5ad_filename)
                with open(target_path, 'wb') as target:
                    target.write(f.read())
                h5ad_path = target_path
        except (IOError, OSError) as e:
            return jsonify({"error": f"Cannot read h5ad file: {str(e)}"}), 400
    else:
        return jsonify({"error": "No h5ad file provided"}), 400

    # Handle csv file similarly
    csv_path = None
    if 'csv' in request.files:
        csv_file = request.files['csv']
        file_config.original_csv_filename = csv_file.filename
        file_config.csv_filename = secure_filename(csv_file.filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file_config.csv_filename)
        csv_file.save(csv_path)
    elif 'csv_path' in request.form:
        csv_path = request.form['csv_path']
        file_config.original_csv_filename = os.path.basename(csv_path)
        file_config.csv_filename = secure_filename(os.path.basename(csv_path))
        try:
            with open(csv_path, 'rb') as f:
                target_path = os.path.join(app.config['UPLOAD_FOLDER'], file_config.csv_filename)
                with open(target_path, 'wb') as target:
                    target.write(f.read())
                csv_path = target_path
        except (IOError, OSError) as e:
            csv_path = None
    
    # Generate CSV if needed
    if not csv_path:
        markers_filename = f"markers_{file_config.h5ad_filename.split('.')[0]}.csv"
        file_config.csv_filename = markers_filename
        csv_path = generate_csv_from_h5ad(original_grouping, h5ad_path, markers_filename)

    return jsonify({
        "message": "Files processed successfully",
        "h5ad_file": file_config.h5ad_filename,
        "csv_file": file_config.csv_filename
    }), 200

def generate_csv_from_h5ad(original_grouping,h5ad_path,csv_filename):
    adata = sc.read_h5ad(h5ad_path)
    sc.tl.rank_genes_groups(adata, original_grouping, method='t-test', n_genes=10)  
    print("DEBUG: generate csv")
    marker_genes_dict = sc.get.rank_genes_groups_df(adata, group=None)
    results = []
    for group in adata.uns['rank_genes_groups']['names'].dtype.names:
        for gene, avg_log2FC in zip(adata.uns['rank_genes_groups']['names'][group], adata.uns['rank_genes_groups']['logfoldchanges'][group]):
            results.append({
                'cluster': group,
                'gene': gene,
                'avg_log2FC': avg_log2FC
            })
    marker_genes_df = pd.DataFrame(results)
    marker_genes_df['avg_log2FC'] = 1
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    marker_genes_df.to_csv(csv_path, index=False)
    return csv_path

@app.route('/dotplot/<filename>', methods=['GET'])
def get_dotplot(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/png')

@app.route('/dotplot/list', methods=['GET'])
def list_dotplots():
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.png') and 'dotplot' in f]
    return jsonify(files)

@app.route('/h5ad/<filename>', methods=['GET'])
def get_h5ad(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, mimetype='application/x-hdf')

@app.route('/umap/<filename>', methods=['GET'])
def get_umap(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/png')

@app.route('/umap/list', methods=['GET'])
def list_umaps():
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.png') and 'umap' in f]
    return jsonify(files)

@app.route('/process', methods=['POST'])
def process_data():
    global hypothesis_agent, experiment_agent, environment_agent, evaluation_agent
    global current_iteration, no_gene_cluster, failed_genes, successful_genes, annotation_dict, analysis_history, annotation_dict_list
    global stabilized_annotation_dict,initial_hypothesis
    global optional_rules_contamination, optional_rules_duplet, adata
    global original_grouping,dotplot
    data = request.json
    stage = data.get('stage', 'hypothesis')
    feedback = data.get('feedback')
    input_dir = UPLOAD_FOLDER
    output_dir = OUTPUT_FOLDER
    h5ad_file = file_config.h5ad_filename
    markers_file = file_config.csv_filename
    try:
        h5ad_dir, h5ad_file = get_current_adata_path(h5ad_file)
        adata = sc.read_h5ad(os.path.join(h5ad_dir, h5ad_file))
        is_zoomed = 'subset_' in h5ad_file
        groupby_param = 'cellmaster_grouping' if is_zoomed else original_grouping
        adata.obs[groupby_param] = adata.obs[groupby_param].astype('category')
        print('[DEBUG] Groupby parameter set to: {}'.format(groupby_param))
        
        if stage == 'hypothesis':
            print('[DEBUG] Hypothesis stage called')
            current_iteration += 1
            hypothesis = data.get('hypothesis', '')
            initial_hypothesis = hypothesis
            h5ad_complete_dir = os.path.join(h5ad_dir, h5ad_file)
            markers_file_complete_dir = os.path.join(UPLOAD_FOLDER, markers_file)
            print('[DEBUG] Hypothesis Agent reading h5ad file from {} and markers file from {}'.format(h5ad_complete_dir, markers_file_complete_dir))
            hypothesis_agent = LiverHypothesisAgent(hypothesis=hypothesis, h5ad_file=h5ad_complete_dir, csv_file=markers_file_complete_dir)
            print('[DEBUG] Hypothesis stage identify_top_gene')
            hypothesis_agent.identify_top_genes()
            print('[DEBUG] Refine Hypothesis')
            refined_hypothesis = hypothesis_agent.refine_hypothesis(annotation_dict=stabilized_annotation_dict, evaluation_result=None, no_gene_cluster=no_gene_cluster, iteration_summary=None)
            message = {"stage": "hypothesis", "result": refined_hypothesis}
            print('[DEBUG] Hypothesis stage completed successfully')

        elif stage == 'experiment':
            print('[DEBUG] Experiment stage called')
            # If there's feedback, adjust the hypothesis first
            refined_hypothesis = None
            if feedback:
                hypothesis_agent.adjust_hypothesis(feedback)
                feedback = None
                print('[DEBUG] Hypothesis adjusted: {}'.format(feedback))
            refined_hypothesis = hypothesis_agent.get_refined_hypothesis()
            
            experiment_agent = LiverExperimentAgent(refined_hypothesis)
            if current_iteration == 0:
                print('[DEBUG] Experiment stage not using annotation_dict')
                experiment_agent.propose_experiment()
            else:
                print('[DEBUG] Experiment stage using annotation_dict')
                experiment_agent.propose_experiment(annotation_dict,stabilized_annotation_dict, no_gene_cluster, failed_genes, successful_genes)
            experiment_proposal = experiment_agent.get_experiment_proposal()
            message = {
                "stage": "experiment",
                "result": experiment_proposal,
                "adjusted_hypothesis": refined_hypothesis
            }
            print('[DEBUG] Experiment stage completed successfully')

        elif stage == 'environment':
            print('[DEBUG] Environment stage called with groupby param: {}'.format(groupby_param))
            experiment_proposal = None
            if feedback:
                experiment_agent.adjust_experiment(feedback)
                feedback = None
                print('[DEBUG] Experiment proposal adjusted: {}'.format(feedback))
            experiment_proposal = experiment_agent.get_experiment_proposal()
            print('[DEBUG] Environment stage get proposal')
            environment_agent = LiverEnvironmentAgent(h5ad_dir, output_dir, h5ad_file)
            existing_genes,dotplot = environment_agent.run_experiment(experiment_proposal, current_iteration, groupby=groupby_param)
            dotplot_filename = f"{current_iteration}_01-marker_dotplot.png"
            message = {
                "stage": "environment",
                "result": existing_genes,
                "dotplot": dotplot_filename,
                "adjusted_experiment": experiment_proposal
            }
            print('[DEBUG] Environment stage completed successfully')

        elif stage == 'optionalRulesDuplet':
            print('[DEBUG] optionalRulesDuplet stage called')
            optional_rule_duplet = feedback
            global dup
            dup = feedback
            message = {"stage": "optionalRulesDuplet", "result": optional_rule_duplet}
            print('[DEBUG] optionalRulesDuplet stage completed successfully. Optional rules duplet: {}'.format(optional_rule_duplet))
            
        elif stage == 'optionalRulesContamination':
            print('[DEBUG] optionalRulesContamination stage called')
            optional_rules_contamination = feedback
            global contam 
            contam = feedback
            message = {"stage": "optionalRulesContamination", "result": "No result for this stage"}
            print('[DEBUG] optionalRulesContamination stage completed successfully. Optional rules contamination: {}'.format(optional_rules_contamination))

        elif stage == 'evaluation':
            print('[DEBUG] Evaluation stage called with duplet_rule and contamination_rule: {}, other feedback: {}'.format(dup, contam))
            evaluation_agent = LiverEvaluationAgent(hypothesis_agent.get_refined_hypothesis(), output_dir, h5ad_dir, environment_agent.existing_genes, h5ad_file)
            possible_cell_types = experiment_agent.get_experiment_proposal()
            print("DEBUG: current iteration eval is",current_iteration)
            evaluation_results = evaluation_agent.evaluate(groupby=groupby_param,dotplot=dotplot,duplet_rule=dup, contamination_rule=contam,possible_cell_types=possible_cell_types,iteration = current_iteration)
            print("DEBUG: EXITING INTO EVALUATE")
            evaluation, failed_genes, successful_genes, marker_genes, empty_keys, similar_clusters_dict = evaluation_results
            #adata = sc.read_h5ad(os.path.join(h5ad_dir, h5ad_file))
            #adata.obs[groupby_param] = adata.obs[groupby_param].astype('category')
            no_gene_cluster = evaluation_agent.find_no_gene_cluster(evaluation)
            prediction = evaluation_agent.prediction(evaluation)
            print('[DEBUG] Prediction generated:\n',prediction)
            annotation_dict = evaluation_agent.execution(prediction)
            print("stabilized annotation: \n",stabilized_annotation_dict)
            if stabilized_annotation_dict:
                new_dict = stabilized_annotation_dict.copy()
                new_dict.update({k: v for k, v in annotation_dict.items() if k not in stabilized_annotation_dict})
                annotation_dict = new_dict
            #stabilized_annotation_dict
            print('[DEBUG] Annotation dictionary generated: {}'.format(annotation_dict))
            umap_filename = generate_umap(current_iteration, is_zoomed, annotation_dict, groupby_param)
            adata.write(os.path.join(h5ad_dir, h5ad_file))
            print('[DEBUG] Updated main h5ad file written to {}'.format(h5ad_dir))
            result = {"evaluation": evaluation, "failed_genes": failed_genes, "successful_genes": successful_genes, "marker_genes": marker_genes, "empty_keys": empty_keys, "similar_clusters_dict": similar_clusters_dict, "no_gene_cluster": no_gene_cluster, "annotation_dict": annotation_dict}
            message = {"stage": "evaluation", "result": result, "umap": umap_filename, "h5ad": h5ad_file}
            print('[DEBUG] Evaluation stage completed successfully')

        else:
            return jsonify({"error": "Invalid stage"}), 400
        
        analysis_history.append(message)
        return jsonify(message)

    except Exception as e:
        return jsonify({"error": f"An error occurred during the {stage} stage: {str(e)}"}), 500

@app.route('/submit-clusters', methods=['POST'])
def submit_clusters():
    global stabilized_annotation_dict,annotation_dict
    print("[DEBUG] submit clusters\n",annotation_dict,stabilized_annotation_dict)
    data = request.json
    selected_clusters = data.get('selectedClusters', [])
    print("[DEBUG] selected list\n",selected_clusters)
    if not selected_clusters:
        return jsonify({'error': 'No clusters provided'}), 400
    stabilized_annotation_list=selected_clusters 
    stabilized_annotation_list = [int(i) for i in stabilized_annotation_list]
    print("[DEBUG] stabilize list\n",stabilized_annotation_list)
    stabilized_annotation_dict = {key: annotation_dict[key] for key in stabilized_annotation_list if key in annotation_dict}
    print("[DEBUG] stabilized_annotation updated")
    print(stabilized_annotation_dict)
    return jsonify({'message': 'Clusters successfully updated', 'stabilized_annotation_dict': stabilized_annotation_dict})

@app.route('/auto-fill-in', methods=['POST'])
def auto_fill_in():
    try:
        umap_file_name,annotation_dict = perform_auto_fill_in()
        print("[DEBUG]",umap_file_name,annotation_dict)
        return jsonify({"message": "auto fill successful", "umap": umap_file_name,  "newClusterInfo": annotation_dict}), 200
    except Exception as e:
        return jsonify({'error auto fill in': str(e)}), 500

def perform_auto_fill_in():
    global stabilized_annotation_dict,annotation_dict,initial_hypothesis
    global adata

    h5ad_dir, h5ad_file = get_current_adata_path(file_config.h5ad_filename)
    adata = sc.read_h5ad(os.path.join(h5ad_dir, h5ad_file))
    adata.obs[original_grouping] = adata.obs[original_grouping].astype('category')
    is_zoomed = 'subset_' in h5ad_file
    print("AutoFillIn function is being executed")

    final_annotation = solve_auto_fill_in(info=initial_hypothesis,input_dir=h5ad_dir,h5ad_file=h5ad_file,original_grouping=original_grouping)
    print("DEBUG: \n",final_annotation)
    if final_annotation:
        merged_dict = {**final_annotation, **annotation_dict}
    print('[DEBUG] merged_dict dictionary generated',merged_dict)
    annotation_dict = merged_dict
    umap_filename = generate_umap(current_iteration, is_zoomed, annotation_dict, original_grouping)
    adata.write(os.path.join(h5ad_dir, h5ad_file))
    annotation_dict = {str(i): label for i, label in enumerate(adata.obs[f"{original_grouping}_labels"].cat.categories)}
    print('[DEBUG] perform fill in ended')
    return umap_filename,annotation_dict

@app.route('/zoom-in', methods=['POST'])
def zoom_in():
    print('[DEBUG] Zoom-in stage called')
    global adata, annotation_dict, stabilized_annotation_dict
    data = request.json
    selected_clusters = data.get('selectedClusters', [])
    resolution = data.get('resolution', 0.1)  # Get resolution from request, default to 0.1  
    stabilized_annotation_dict={}
    if not selected_clusters:
        return jsonify({"error": "No clusters selected for zoom-in"}), 400
    
    try:
        h5ad_dir, h5ad_file = get_current_adata_path(file_config.h5ad_filename)
        adata = sc.read_h5ad(os.path.join(h5ad_dir, h5ad_file))
        selected_clusters = [int(c) for c in selected_clusters]
        adata_subset = adata[adata.obs[original_grouping].isin(selected_clusters)].copy()
        print('[DEBUG] subset selected clusters: {} and store in adata_subset[cellmaster_grouping]'.format(selected_clusters))
        
        # perform clustering
        sc.pp.neighbors(adata_subset)
        sc.tl.leiden(adata_subset, flavor="leidenalg", n_iterations=2, resolution=resolution)
        # copy leiden to cellmaster_grouping
        adata_subset.obs['cellmaster_grouping'] = adata_subset.obs['leiden']
        # initialize cellmaster_grouping_labels with numerical labels
        adata_subset.obs['cellmaster_grouping_labels'] = adata_subset.obs['cellmaster_grouping'].astype(str)
        print('[DEBUG] cellmaster_grouping_labels initialized')
        
        # create umap
        sc.pl.umap(adata_subset, color="cellmaster_grouping", size=2, legend_loc="on data", show=False)
        zoomed_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith(f"{current_iteration}_") and 'zoomed' in f]
        umap_filename = f"{current_iteration}_{len(zoomed_files)}_zoomed_umap_plot.png"
        plt.savefig(os.path.join(OUTPUT_FOLDER, umap_filename))
        print('[DEBUG] zoomed umap plot saved as {}'.format(umap_filename))
        plt.close()
        subset_filename = f"subset_{current_iteration}.h5ad"
        adata_subset.write(os.path.join(OUTPUT_FOLDER, subset_filename))
        markers_file = generate_markers_csv(OUTPUT_FOLDER, OUTPUT_FOLDER, subset_filename, groupby='cellmaster_grouping')
        if not markers_file:
            print('[WARNING] Failed to generate markers file for subset')
        print('[DEBUG] subset h5ad file written to {}'.format(OUTPUT_FOLDER))
        adata.write(os.path.join(UPLOAD_FOLDER, file_config.h5ad_filename))
        annotation_dict = None
        print('[DEBUG] annotation_dict set to None')
        
        # return cluster info with cellmaster_grouping_labels
        new_cluster_info = adata_subset.obs['cellmaster_grouping_labels'].value_counts().to_dict()
        return jsonify({"message": "Zoom-in successful", "umap": umap_filename, "selectedClusters": selected_clusters, "newClusterInfo": new_cluster_info}), 200
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during zoom-in: {str(e)}"}), 500

@app.route('/zoom-out', methods=['POST'])
def zoom_out():
    global adata, current_iteration, annotation_dict,stabilized_annotation_dict
    
    try:
        # Load the full dataset and the subset
        print('[DEBUG] start zoom out')
        h5ad_dir, h5ad_file = get_current_adata_path(file_config.h5ad_filename)
        adata_subset = sc.read_h5ad(os.path.join(h5ad_dir, h5ad_file))
        print('[DEBUG] current h5ad path: {}'.format(os.path.join(h5ad_dir, h5ad_file)))
        
        adata = sc.read_h5ad(os.path.join(UPLOAD_FOLDER, file_config.h5ad_filename))
        print('[DEBUG] successfully load subset adata')
        
        # Ensure required columns exist
        if f"{original_grouping}_labels" not in adata.obs.columns:
            print(f'[DEBUG] Creating {original_grouping}_labels')
            adata.obs[f"{original_grouping}_labels"] = adata.obs[original_grouping].astype(str)
        
        if 'cellmaster_grouping_labels' not in adata_subset.obs.columns:
            print('[WARNING] cellmaster_grouping_labels not found in subset, using cellmaster_grouping')
            adata_subset.obs['cellmaster_grouping_labels'] = adata_subset.obs['cellmaster_grouping'].astype(str)
        
        # Merge the subset data back into the main dataset
        print('[DEBUG] before merge main adata shape: {}'.format(adata.shape))
        print('[DEBUG] before merge adata shape: {}'.format(adata_subset.shape))
        adata = merge_subset_to_main(adata, adata_subset, f"{original_grouping}_labels", 'cellmaster_grouping_labels', 'merged_prediction')
        print('[DEBUG] after merge main adata shape: {}'.format(adata.shape))
        
        # clean the subset h5ad file
        for file in os.listdir(OUTPUT_FOLDER):
            if file.endswith(".h5ad"):
                os.remove(os.path.join(OUTPUT_FOLDER, file))
                
        # write back to main h5ad file
        adata.write(os.path.join(UPLOAD_FOLDER, file_config.h5ad_filename))
        # Generate new markers for the full dataset
        #markers_file = generate_markers_csv(UPLOAD_FOLDER, UPLOAD_FOLDER, file_config.h5ad_filename)
        #if not markers_file:
        #    print('[WARNING] Failed to generate markers file after zoom-out')
        # Clean up generated subsets
        for file in os.listdir(OUTPUT_FOLDER):
            if file.endswith(".h5ad"):
                os.remove(os.path.join(OUTPUT_FOLDER, file))
        
        # Generate new UMAP for the full dataset
        umap_filename = generate_umap(current_iteration, False, adata.obs['merged_prediction'].to_dict(), groupby='merged_prediction')
        
        # Generate annotation_dict based on original_grouping_labels
        original_grouping_labels = adata.obs[f"{original_grouping}_labels"]
        annotation_dict = {int(i): label for i, label in enumerate(original_grouping_labels.cat.categories)}
        print("[DEBUG] Annotation_dict after zoom out: ",annotation_dict)
        print('[DEBUG] Generated annotation_dict:', annotation_dict)
        return jsonify({
            "message": "Zoom-out successful",
            "umap": umap_filename,
            "newClusterInfo": annotation_dict
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during zoom-out: {str(e)}"}), 500
      
@app.route('/reset', methods=['POST'])
def reset():
    global file_config
    global hypothesis_agent, experiment_agent, environment_agent, evaluation_agent
    global current_iteration, no_gene_cluster, failed_genes, successful_genes
    global annotation_dict, analysis_history, annotation_dict_list
    global optional_rules_contamination, optional_rules_duplet, adata
    global stabilized_annotation_dict, initial_hypothesis, original_grouping
    
    # reset file_config
    file_config = FileConfig()
    
    # Reset all global variables
    hypothesis_agent = None
    experiment_agent = None
    environment_agent = None
    evaluation_agent = None
    current_iteration = 0
    no_gene_cluster = []
    failed_genes = []
    successful_genes = []
    annotation_dict = None
    annotation_dict_list = []
    analysis_history = []
    optional_rules_contamination = None
    optional_rules_duplet = None
    adata = None
    stabilized_annotation_dict = {}
    initial_hypothesis = ""
    original_grouping = ''
    
    # Clean up output files
    for folder in [OUTPUT_FOLDER, UPLOAD_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
                
    # refresh the page
    
    
    return jsonify({"message": "Application state reset successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)