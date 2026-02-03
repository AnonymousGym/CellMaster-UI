import os
import re
import ast
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import time
from agents.environment_agent.base_environment_agent import BaseEnvironmentAgent

matplotlib.use('Agg')

class LiverEnvironmentAgent(BaseEnvironmentAgent):
    def __init__(self, input_dir, output_dir, file_name):
        super().__init__(simulation_environment='liver', input_dir=input_dir, output_dir=output_dir)
        self.adata = ad.read_h5ad(os.path.join(input_dir, file_name))
        self.existing_genes = []  # Initialize the attribute

    def extract_experiment_values(self, experiment_proposal):
        patterns = [
            r'\*\*Python List of Marker Genes:\*\*\s*(\[.*?\])',
            r'MARKER_GENES\s*=\s*(\[.*?\])',
            r'marker genes:\s*(\[.*?\])',
            r'ALL_MARKERS\s*=\s*(\[.*?\])'
        ]
        for pattern in patterns:
            match = re.search(pattern, experiment_proposal, re.DOTALL)
            if match:
                genes_list_str = match.group(1)
                try:
                    genes_list = ast.literal_eval(genes_list_str)
                    return [gene.capitalize() for gene in genes_list if isinstance(gene, str)]
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating the list: {e}")
        print("No gene list found in the proposal.")
        return []

    def filter_existing_genes(self, genes,scanpy):
        if scanpy:
            return [gene for gene in genes if gene.lower() in map(str.lower, self.adata.var_names)]
        else:
            return [gene for gene in genes if gene in map(str, self.adata.var_names)]
    
    def run_experiment(self, experiment_proposal, iteration,groupby="seurat_clusters",scanpy=False):
        genes = self.extract_experiment_values(experiment_proposal)    
        self.existing_genes = self.filter_existing_genes(genes,scanpy=True)
        lower_to_original = {name.lower(): name for name in self.adata.var_names}
        found_genes = []
        for gene in self.existing_genes:
            if gene in self.adata.var_names:
                found_genes.append(gene)
            elif gene.lower() in lower_to_original:
                found_genes.append(lower_to_original[gene.lower()])
            else:
                raise EOFError("gene not found: ",gene)
        self.existing_genes=found_genes
        print("Existing genes:", self.existing_genes)
        if self.existing_genes:
            if scanpy:
                self.existing_genes = [gene.upper() for gene in self.existing_genes]
            try:
                plt.figure(figsize=(20, 20))  # Increase figure size
                sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, show=False)
                plt.tight_layout()
                dotplot_name = str(iteration)+'_01-marker_dotplot.png'
                plt.savefig(os.path.join(self.output_dir, dotplot_name), dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free up memory
                dotplot = sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, return_fig=True)
            except Exception as e:
                print(f"Error creating dotplot: {e}")
        else:
            print("No existing genes found. Skipping dotplot creation.")
        
        self.existing_genes = list(set(self.existing_genes))
        return self.existing_genes,dotplot
    

    def run_dotplot(self, marker_gene_list, iteration,groupby="seurat_clusters",species="mouse"):
        match = re.search(r"\[.*\]", marker_gene_list, re.DOTALL)
        if match:
            lst = ast.literal_eval(match.group())  # Convert string to list
        self.existing_genes = [gene for gene in lst if gene.lower() in map(str.lower, self.adata.var_names)]
        self.existing_genes = list(set(self.existing_genes))
        if self.existing_genes:
            if species == "human":
                self.existing_genes = [gene.upper() for gene in self.existing_genes]
            try:
                plt.figure(figsize=(20, 20))  # Increase figure size
                sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, show=False)
                plt.tight_layout()
                dotplot_name = str(iteration)+'_01-marker_dotplot.png'
                plt.savefig(os.path.join(self.output_dir, dotplot_name), dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free up memory
                dotplot = sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, return_fig=True)
            except Exception as e:
                print(f"Error creating dotplot: {e}")
        else:
            print("No existing genes found. Skipping dotplot creation.")
        return self.existing_genes,dotplot
    