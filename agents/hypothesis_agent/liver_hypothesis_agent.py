import base64
import os
from agents.hypothesis_agent.base_hypothesis_agent import BaseHypothesisAgent
import openai
from config.settings import OPENAI_API_KEY
import anndata as ad
from utils.liver_process_toolkit import get_top_differential_genes
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import requests
from bs4 import BeautifulSoup
import re
import scanpy as sc

openai.api_key = OPENAI_API_KEY

class LiverHypothesisAgent(BaseHypothesisAgent):
    def __init__(self, hypothesis, h5ad_file, csv_file=None):
        super().__init__(hypothesis)
        self.h5ad_file = h5ad_file
        if csv_file:
            self.marker_name = csv_file
        self.adata = ad.read_h5ad(self.h5ad_file)
        self.top_genes = None
        self.cluster_images = []
        self.iteration = 0
        self.reference_dict = None
        
    def get_ensembl_id(self, gene_name):
        server = "https://rest.ensembl.org"
        ext = f"/xrefs/symbol/homo_sapiens/{gene_name}?"
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.get(server + ext, headers=headers)
            response.raise_for_status()
            data = response.json()
            print("Fetching Ensembl ID successfully: ", data)
            for entry in data:
                if entry["type"] == "gene":
                    return entry["id"]
        except requests.RequestException as e:
            print(f"Error fetching Ensembl ID for {gene_name}: {str(e)}")
        return None
            
    def identify_top_genes(self,type=None):
        n_genes = 5 if self.iteration == 0 else 3
        if type == "scanpy":
            self.top_genes = get_top_differential_genes(self.marker_name, n_genes=n_genes,cluster="group",foldchange="logfoldchanges",gene="names")
        else:
            self.top_genes = get_top_differential_genes(self.marker_name, n_genes=n_genes)
        reference_clusters = ["Hepatocytes", "T-cells", "Kupffer cells", "Plasma cells", "Hepatocytes", "Endothelial cells", "Kupffer cells", "Smooth muscle cells", "T-cells", "Hepatocytes", "Kupffer cells", "Fibroblasts", "Hepatocytes", "T-cells", "T-cells", "Kupffer cells", "B-cells", "Cholangiocytes", "Hepatocytes", "Erythroid cells"]
        '''
        reference_expression_dict = {}
        print(f"Fetching {len(self.top_genes)} gene lists...")
        for key, gene_list in self.top_genes.items():
            print(f"Fetching {key} gene list, there are {len(gene_list)} genes ...")
            for gene in gene_list:
                ensembl_id = self.get_ensembl_id(gene)
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
                    key_name = "Cluster "+str(key)+" "+gene
                    reference_expression_dict[key_name] = expression_dict
        self.reference_dict = reference_expression_dict
        '''
    def top_gene_from_adata(self):
        sc.tl.rank_genes_groups(self.adata, 'leiden', method='wilcoxon')
        top_genes = {}
        for cluster in self.adata.obs['leiden'].unique():
            # Convert cluster to string if necessary
            cluster_str = str(cluster)
            cluster_genes = sc.get.rank_genes_groups_df(self.adata, group=cluster_str).head(10)
            top_genes[cluster] = list(cluster_genes["names"])
        self.top_genes = top_genes
        return top_genes
    
    def refine_hypothesis(self, annotation_dict=None,evaluation_result=None, no_gene_cluster=None,iteration_summary = None):
        self.iteration += 1

        if evaluation_result:
            # Filter top genes based on no_gene_cluster and failed_genes
            filtered_top_genes = {}
            for cluster, genes in self.top_genes.items():
                if cluster not in no_gene_cluster:
                    filtered_genes = [gene for gene in genes]
                    if filtered_genes:
                        filtered_top_genes[cluster] = filtered_genes
            self.top_genes = filtered_top_genes

        #content = f"Literature Summary:\n\n{self.summary}\n\n"
        content = f"Top {len(self.top_genes)} differentially expressed genes: {self.top_genes}\n\n"
        if self.reference_dict:
            content += f"You can refer to the possible cell types of these top genes in this dictionary{self.reference_dict}"
        content += f"Current Hypothesis:\n{self.hypothesis}\n\n"
        if annotation_dict:
            content += f"The cell type annotation from previous iterations {annotation_dict}"
        if no_gene_cluster:
            content += f"Clusters without need to be focused on: {no_gene_cluster}\n\n"
        #if evaluation_result:
        #    content += f"Evaluation Result:\n{evaluation_result}\n"
        if iteration_summary:
            content += f"This is summary of previous iteration annotation, with information of next steps to take. {iteration_summary}"

        model = "gpt-4o"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in cell biology. Based on top differentially expressed genes, previous cell type annotation (if provided), Clusters without need to be focused on (if provided), summary of previous iteration annotation (if provided), and failed genes (if provided), refine the given hypothesis to be more accurate and specific."},
                {"role": "user", "content": content}
            ]#,temperature=0
        )
        self.refined_hypothesis = response.choices[0].message['content'].strip()
        #self.hypothesis = self.refined_hypothesis
        return self.refined_hypothesis
        
    def adjust_hypothesis(self, user_feedback):
        content = f"Refined Hypothesis:\n\n{self.refined_hypothesis}\n\nUser Feedback:\n\n{user_feedback}\n\n"
        content += "Please adjust the refined hypothesis based on the user feedback, focusing on the tissue-related insights."

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in biology. Adjust the refined hypothesis based on the user feedback. If no user feedback is provided, return the refined hypothesis with the exact wording."},
                {"role": "user", "content": content}
            ],
            temperature=0
        )
        self.refined_hypothesis = response.choices[0].message['content'].strip()

    def summarize_difference(self):
        content = f"Refined Hypothesis:\n\n{self.refined_hypothesis}\n\nPrevious Hypothesis:\n\n{self.hypothesis}\n\n"
        content += "Please compare the two versions of hypthesis and summarize the difference"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in biology. Adjust the refined hypothesis based on the user feedback. If no user feedback is provided, return the refined hypothesis with the exact wording."},
                {"role": "user", "content": content}
            ]#,temperature=0
        )
        reply = response.choices[0].message['content'].strip()
        return reply
        
    def run_arxiv_process(self):
        self.generate_search_query()
        self.retrieve_literature()
        self.summarize_literature()
        #self.identify_top_genes()

    def recommend_celltype(self,gene):
        reference_clusters = ["Hepatocytes", "T-cells", "Kupffer cells", "Plasma cells", "Hepatocytes", "Endothelial cells", "Kupffer cells", "Smooth muscle cells", "T-cells", "Hepatocytes", "Kupffer cells", "Fibroblasts", "Hepatocytes", "T-cells", "T-cells", "Kupffer cells", "B-cells", "Cholangiocytes", "Hepatocytes", "Erythroid cells"]

        ensembl_id = self.get_ensembl_id(gene)
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