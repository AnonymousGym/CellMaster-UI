import os
import anndata as ad
from matplotlib import pyplot as plt
import pandas as pd
import re
import json
from scipy.stats import zscore
import numpy as np
import scanpy as sc

from config.settings import OPENAI_API_KEY

def get_top_differential_genes(marker_file, n_genes=10,cluster="cluster",foldchange = 'avg_log2FC',gene="gene"):
    df = pd.read_csv(marker_file)
    sorted_df = df.sort_values(by=[cluster, foldchange], ascending=[True, False])
    top_genes_dict = sorted_df.groupby(cluster).apply(lambda x: x.head(n_genes)[gene].tolist()).to_dict()
    return top_genes_dict

def truncate_eval(text):
    start = text.find('{')
    end = text.rfind('}') + 1
    json_str = text[start:end]

    # Print the extracted dictionary
    print(json_str)  

import requests
from bs4 import BeautifulSoup
def get_ensembl_id(gene_name):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/homo_sapiens/{gene_name}?"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get(server + ext, headers=headers)
        response.raise_for_status()
        data = response.json()
        for entry in data:
            if entry["type"] == "gene":
                return entry["id"]
    except requests.RequestException as e:
        print(f"Error fetching Ensembl ID for {gene_name}: {str(e)}")
    return None
  
def recommend_celltype(gene):
    reference_clusters = ["Hepatocytes", "T-cells", "Kupffer cells", "Plasma cells", "Hepatocytes", "Endothelial cells", "Kupffer cells", "Smooth muscle cells", "T-cells", "Hepatocytes", "Kupffer cells", "Fibroblasts", "Hepatocytes", "T-cells", "T-cells", "Kupffer cells", "B-cells", "Cholangiocytes", "Hepatocytes", "Erythroid cells"]

    ensembl_id = get_ensembl_id(gene)
    if ensembl_id:
        url = f"https://www.proteinatlas.org/{ensembl_id}-{gene}/single+cell+type/liver"
        response = requests.get(url)
        html_content = response.content
        with open('source.html', 'wb') as file:
            file.write(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        pattern = re.compile(r'group\d+ c-\d+')
        matching_rows = soup.find_all('tr', class_=pattern)
        value_list = []
        count = 0
        for row in matching_rows:
            row_string = str(row)  
            match = re.search(r'Expression:\s*(\d+\.\d+)\s*nTPM', row_string)
            if match:
                number = match.group(1)
                value_list.append(number)
            else:
                value_list.append(0)
                print("No match found in cluster: ",count)
            count += 1
        expression_dict = {}
        if len(value_list) > 0:
            expression_dict = dict(zip(reference_clusters, value_list))
    max_key = max(expression_dict, key=lambda k: float(expression_dict[k]))
    return max_key

def identify_marker_genes(dotplot_data, dotplot_data_frac, exp_thresh=0.5, frac_thresh=0.5):
    """
    Identify potential marker genes for each cluster based on expression and fraction thresholds.

    Args:
    - dotplot_data: DataFrame of average expression levels.
    - dotplot_data_frac: DataFrame of fraction of cells expressing each gene.
    - exp_thresh: Minimum average expression level to consider a gene as a marker.
    - frac_thresh: Minimum fraction of cells required to express a gene for it to be a marker.

    Returns:
    - marker_genes: Dictionary where keys are clusters and values are lists of marker genes.
    """
    marker_genes = {}

    for cluster in dotplot_data.index:
        marker_genes[cluster] = []
        for gene in dotplot_data.columns:
            if dotplot_data.loc[cluster, gene] >= exp_thresh and dotplot_data_frac.loc[cluster, gene] >= frac_thresh:
                marker_genes[cluster].append(gene)

    return marker_genes

def zscore_normalize_expression(dotplot_data):
    """
    Apply Z-score normalization to gene expression levels.

    Args:
    - dotplot_data: DataFrame of average expression levels.

    Returns:
    - normalized_data: DataFrame of Z-score normalized expression levels.
    """
    normalized_data = dotplot_data.apply(zscore, axis=0)  # Normalize each gene column-wise
    return normalized_data

def min_max_scale_expression(dotplot_data):
    """
    Apply min-max scaling to gene expression levels.

    Args:
    - dotplot_data: DataFrame of average expression levels.

    Returns:
    - scaled_data: DataFrame of min-max scaled expression levels.
    """
    scaled_data = dotplot_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    return scaled_data

def compute_logfc(dotplot_data, cluster1, cluster2):
    """
    Compute log-fold change of gene expression between two clusters.

    Args:
    - dotplot_data: DataFrame of average expression levels.
    - cluster1: First cluster for comparison.
    - cluster2: Second cluster for comparison.

    Returns:
    - logfc: Series of log-fold change values for all genes.
    """
    logfc = np.log2(dotplot_data.loc[cluster1] + 1) - np.log2(dotplot_data.loc[cluster2] + 1)
    return logfc

def identify_distinguishing_markers(dotplot_data, dotplot_data_frac, cluster1, cluster2, exp_thresh=0.5, frac_thresh=0.5, logfc_thresh=1.0):
    """
    Identify marker genes that can distinguish between two clusters.

    Args:
    - dotplot_data: DataFrame of average expression levels.
    - dotplot_data_frac: DataFrame of fraction of cells expressing each gene.
    - cluster1: First cluster for comparison.
    - cluster2: Second cluster for comparison.
    - exp_thresh: Minimum average expression level to consider a gene as a marker.
    - frac_thresh: Minimum fraction of cells required to express a gene for it to be a marker.
    - logfc_thresh: Minimum log-fold change required for a gene to be considered distinguishing.

    Returns:
    - distinguishing_markers: List of genes that distinguish between the two clusters.
    """
    logfc = compute_logfc(dotplot_data, cluster1, cluster2)
    
    distinguishing_markers = []
    
    for gene in dotplot_data.columns:
        if (dotplot_data.loc[cluster1, gene] >= exp_thresh or dotplot_data.loc[cluster2, gene] >= exp_thresh) and \
           (dotplot_data_frac.loc[cluster1, gene] >= frac_thresh or dotplot_data_frac.loc[cluster2, gene] >= frac_thresh) and \
           abs(logfc[gene]) >= logfc_thresh:
            distinguishing_markers.append(gene)
    
    return distinguishing_markers

def find_similar_cluster_pairs(dotplot_data, dotplot_data_frac, exp_diff_thresh=0.5, frac_diff_thresh=0.5, max_diff_genes=2, logfc_thresh=1.0):
    """
    Identify similar cluster pairs where most genes have similar expression profiles,
    but they differ in a small number of genes.

    Args:
    - dotplot_data: DataFrame of average expression levels.
    - dotplot_data_frac: DataFrame of fraction of cells expressing each gene.
    - exp_diff_thresh: Threshold for considering expression levels as similar.
    - frac_diff_thresh: Threshold for considering fraction of expression as similar.
    - max_diff_genes: Maximum number of genes where clusters can differ.
    - logfc_thresh: Minimum log-fold change required for a gene to be considered distinguishing.

    Returns:
    - similar_pairs: List of tuples containing similar cluster pairs and their distinguishing genes.
    """
    clusters = dotplot_data.index
    similar_pairs = []
    
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            diff_genes = identify_distinguishing_markers(dotplot_data, dotplot_data_frac, cluster1, cluster2,
                                                         exp_thresh=exp_diff_thresh,
                                                         frac_thresh=frac_diff_thresh,
                                                         logfc_thresh=logfc_thresh)
            # If the number of differing genes is below or equal to the threshold, consider them similar
            if len(diff_genes) <= max_diff_genes:
                similar_pairs.append((cluster1, cluster2, diff_genes))
    
    return similar_pairs

def plot_2(cell_dict,iteration,input_dir,output_dir,file_name,groupby,output_column_name):
    adata = ad.read_h5ad(f"{input_dir}/{file_name}")
    adata.obs[groupby] = adata.obs[groupby].astype(int)
    adata.obs[output_column_name] = adata.obs[groupby].map(cell_dict)
    plt.figure(figsize=(20, 20))
    sc.pl.umap(adata, color=output_column_name,legend_loc="on data",title='Predicted Cell Types')
    annotation_name = str(iteration)+'_02-prediction.png'
    plt.savefig(os.path.join(output_dir, annotation_name), dpi=300, bbox_inches='tight')
    plt.close()
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    adata.write(f"{input_dir}/{file_name}")

import anndata as ad
import ast

def solve_rest_clusters(merged_dict,no_gene_cluster,h5ad_file,marker_file,info=None,input_dir='data/liver/input',original_grouping="leiden"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    messages = [
        {
            "role": "system",
            "content": "You are expert in scRNA sequencing cell type annotation."
        },
        {
            "role": "user",
            "content": f'''
            look at this dict: {merged_dict}. If there are values that is Unknown, NA or not a cell type, ONLY output the correlated keys as a text list, such as "[1,2]", no other word should exist
        '''
            }
        ]
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 2000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    summary = response.json()
    list_str = summary["choices"][0]["message"]["content"]
    try:
        actual_list= ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        return None
    unsolved_list = list(no_gene_cluster) + actual_list
    if len(unsolved_list) == 0:
        return
    #adata = sc.read_h5ad(os.path.join(input_dir, h5ad_file))
    #sc.tl.rank_genes_groups(adata, original_grouping, method='wilcoxon')
    marker_data = pd.read_csv(os.path.join(input_dir, marker_file))
    top_genes = {}
    for cluster, group in marker_data.groupby('cluster'):
        # Store the gene names in a list for each cluster
        top_genes[cluster] = list(group['gene'])
    subset_top_genes = {key: top_genes[key] for key in unsolved_list if key in top_genes}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    messages = [
        {
            "role": "system",
            "content": "You are expert in scRNA sequencing cell type annotation."
        },
        {
            "role": "user",
            "content": f'''
            this is background information: {info}
            look at this dict: {subset_top_genes}. This is cluster number and the corresponding top differential genes of each cluster. Please provide cell type annotation for each cluster. 
            Output in text dict format just like the input dict. Keys are number of cluster, and Values are strings of cell type names. Output should be text dict, no other word should exist.
        '''
            }
        ]
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 3000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    reply = response.json()["choices"][0]["message"]["content"]
    sanitized_str = reply.replace("```", "")
    try:
        return ast.literal_eval(sanitized_str)
    except (ValueError, SyntaxError):
        return None
    
import warnings

def solve_auto_fill_in(h5ad_file,info=None,input_dir='data/liver/input',original_grouping="leiden"):
    print("solving auto fill in")
    adata = sc.read_h5ad(os.path.join(input_dir, h5ad_file))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sc.tl.rank_genes_groups(adata, original_grouping, method='wilcoxon')
        top_genes = {}
        for cluster in adata.obs[original_grouping].unique():
            # Convert cluster to string if necessary
            cluster_str = str(cluster)
            cluster_genes = sc.get.rank_genes_groups_df(adata, group=cluster_str).head(10)
            top_genes[cluster] = list(cluster_genes["names"])
    subset_top_genes = top_genes
    print("top gene generated for auto cluster")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    messages = [
        {
            "role": "system",
            "content": "You are expert in scRNA sequencing cell type annotation."
        },
        {
            "role": "user",
            "content": f'''
            this is background information: {info}
            look at this dict: {subset_top_genes}. This is cluster number and the corresponding top differential genes of each cluster. Please provide cell type annotation for each cluster. 
            Output in text dict format just like the input dict. Keys are number of cluster, and Values are strings of cell type names. Output should be text dict, no other word should exist.
        '''
            }
        ]
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 2000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    reply = response.json()["choices"][0]["message"]["content"]
    sanitized_str = reply.replace("```", "")
    print("[DEBUG] auto fill in result\n",ast.literal_eval(sanitized_str))
    try:
        return ast.literal_eval(sanitized_str)
    except (ValueError, SyntaxError):
        return None