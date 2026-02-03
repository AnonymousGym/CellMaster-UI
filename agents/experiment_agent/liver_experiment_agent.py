from agents.experiment_agent.base_experiment_agent import BaseExperimentAgent
import openai
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class LiverExperimentAgent(BaseExperimentAgent):
    def propose_experiment(self,annotation_dict = None,stabilized_annotation_dict=None,no_gene_cluster=None,failed_genes=None,successful_genes=None,subset=False):
        if annotation_dict and stabilized_annotation_dict:
            complement = {key: value for key, value in annotation_dict.items() if key not in stabilized_annotation_dict}
        if subset:
            prompt = f'''
    You are a bioinformatics expert specializing in cell annotation. Your task is to propose an experiment for cell type annotation based on the following information:

    Refined hypothesis about this annotation: {self.hypothesis}

    Output format:
    1. List of cell types with their markers:
    [Cell Type 1]: Gene A; Gene B; Gene C
    [Cell Type 2]
    ...

    2. Python list of all marker genes:
    MARKER_GENES = ['GeneA', 'GeneB', 'GeneC', ...]

    Remember: 
    - Be specific and concise in your descriptions.
    - Ensure all cell types have at least 3 marker genes.
    - Include the Python list of all marker genes at the end of your response, don't use any backtick.
    '''
            if no_gene_cluster:
                prompt += f"""
    Instructions:
    0. We have already determined some of clusters, the information is in {stabilized_annotation_dict}
    1. 
    a) The most important unsolved clusters are {no_gene_cluster}.
    b) If applicable, this {complement} is cell types we labeled with lower confidence. We should consider them again.
    2. For the cell types we have already successfully used, the gene marker list is {successful_genes}.
    For large cell types like hepatocyte or B cell, the remaining clusters might still contain them, but for smaller cell types you don't need to think about them again.
    For the marker genes we already used but did not detect any expression, the list is {failed_genes}. So you don't need to try these gene and related cell type again.
    3. Do not specify the cluster with cell type here. You can just output cell type and related marker genes.
    4. Only provide proposal of unlabeled clusters, consider potential overlaps in marker gene expression between cell types:
    a) Name of the cell type
    b) 3-5 marker genes
    """

            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
            {"role": "system", "content": "You are an AI trained to design scientific experiments based on hypotheses and background information."},
            {"role": "user", "content": prompt}
            ],
            )
            self.experiment_proposal = response.choices[0].message['content'].strip()
            return

        if no_gene_cluster:
            prompt = f'''
    You are a bioinformatics expert specializing in liver cell annotation. Your task is to propose an experiment for cell type annotation based on the following information:

    Refined hypothesis: {self.hypothesis}

    Instructions:
    0. We have already determined some of clusters, the information is in {stabilized_annotation_dict}
    1.
    a) The most important unsolved clusters are {no_gene_cluster}.
    b) {complement} is cell types we labeled with lower confidence. We should consider them again.    
    2. For the cell types we have already successfully used, the gene marker list is {successful_genes}.
    For large cell types like hepatocyte or B cell, the remaining clusters might still contain them, but for smaller cell types you don't need to think about them again.
    For the marker genes we already used but did not detect any expression, the list is {failed_genes}. So you don't need to try these gene and related cell type again.
    3. Do not specify the cluster with cell type here. You can just output cell type and related marker genes.
    4. Only provide proposal of unlabeled clusters, consider potential overlaps in marker gene expression between cell types:
    a) Name of the cell type
    b) 3-5 marker genes

    Output format:
    1. List of cell types with their markers:
    [Cell Type 1]: Gene A; Gene B; Gene C
    [Cell Type 2]
    ...

    2. Python list of all marker genes:
    MARKER_GENES = ['GeneA', 'GeneB', 'GeneC', ...]

    Remember: 
    - Be specific and concise in your descriptions.
    - Ensure all cell types have at least 3 marker genes.
    - Include the Python list of all marker genes at the end of your response, don't use any backtick.
    '''

            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
            {"role": "system", "content": "You are an AI trained to design scientific experiments based on hypotheses and background information."},
            {"role": "user", "content": prompt}
            ],
            )

            self.experiment_proposal = response.choices[0].message['content'].strip()
        else:
            print("DEBUG: in propose")
            prompt = f'''
    You are a bioinformatics expert specializing in liver cell annotation. Your task is to propose an experiment for cell type annotation based on the following information:

    Refined hypothesis: {self.hypothesis}

    Instructions:
    1. Specify 10-30 cell types likely to be present in the tissue.
    2. For each cell type, provide:
    a) Name of the cell type
    b) 3-5 marker genes

    3. Consider potential overlaps in marker gene expression between cell types.

    Output format:
    1. List of cell types with their markers:
    [Cell Type 1]: Gene A; Gene B; Gene C
    [Cell Type 2]
    ...

    2. Python list of all marker genes:
    MARKER_GENES = ['GeneA', 'GeneB', 'GeneC', ...]

    Remember: 
    - Be specific and concise in your descriptions.
    - Ensure all cell types have at least 3 marker genes.
    - Include the Python list of all marker genes at the end of your response.
    '''
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
            {"role": "system", "content": "You are an AI trained to design scientific experiments based on hypotheses and background information."},
            {"role": "user", "content": prompt}
            ],
            )
        self.experiment_proposal = response.choices[0].message['content'].strip()
        
    def adjust_experiment(self, user_feedback):
        content = f"Experiment Proposal:\n\n{self.experiment_proposal}\n\nUser Feedback:\n\n{user_feedback}\n\n"
        content += "Please adjust the experiment proposal based on the user feedback, focusing on the tissue-related insights."

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in cell biology. Adjust the experiment proposal based on the user feedback."},
                {"role": "user", "content": content}
            ]#,temperature=0
        )
        self.experiment_proposal = response.choices[0].message['content'].strip()