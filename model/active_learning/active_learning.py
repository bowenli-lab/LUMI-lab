from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import colorcet as cc


SMI_NAME_COL = "smi_name"
NPZ_ARR_INDEX = "arr_0"

class InferenceDataAnalyzer:
    def __init__(self, 
                 result_dir : str,
                 embedding_path: str, 
                 mapping_df_path: str,
                 name_df_path: str,
                 ensemble_num : int = 5,
                 smi_reduction: str = "mean",
                 ensemble_reduction: str = "mean",
                 embedding_reduction: str = "first",
                 dim_reduction_method: str = "umap",
                 use_cuda: bool = True) -> None:
        """
        This is the batch active learner class, which is used to load the ensemble results and embeddings,
        do necessary preprocessing, visualization, and provide the interface for aqcuisition functions.
        
        Args:
            result_dir: str - The directory containing the ensemble results.
            embedding_path: str - The path to the embeddings.
            mapping_df_path: str - The path to the mapping dataframe.
            name_df_path: str - The path to the name dataframe.
            ensemble_num: int - The number of ensemble models.
            smi_reduction: str - The reduction method for the test-time augmentation (TTA) results, e.g.
                                each smi_name has multiple predictions from different conformer (augmentation).
                                The reduction method can be "mean", "median", "max", "min", etc.
            ensemble_reduction: str - The reduction method for the ensemble results, e.g. the ensemble has multiple
                                        models, and the reduction method can be "mean", "median", "max", "min", etc.
            dim_reduction_method: str - The reduction method for the embeddings, e.g. the embeddings can be from different
                                        conformers, and the reduction method can be "first"
        """
        
        self.result_dir = result_dir
        self.ensemble_num = ensemble_num
        self.embedding_path = embedding_path
        self.pred_res_list = self._load_result()
        self.smi_reduction = smi_reduction
        self.ensemble_reduction = ensemble_reduction
        self.embedding_reduction = embedding_reduction
        
        
        self.pred_df, self.smi_reduced_pred_df, \
            self.pred_var, self.ensemble_smi_reduced_pred_df = self._preprocess_result(self.pred_res_list)
        
        self.full_smi_name = self._get_smi_name()
    
        
        self.mapping_table = self._construct_mapping_table(mapping_df_path, name_df_path)
        
        
        self.embeddings, self.embeddings_df = self._load_embedding()
        
        self.cluster_df = self._compute_cluster(use_cuda=use_cuda)
        
        self.dim_reduced_embeddings = self._compute_dim_reduction(reduction_method=dim_reduction_method,
                                                                  use_cuda=use_cuda)
                
        print("InferenceDataAnalyzer initialized.")
        
    def _construct_mapping_table(self,mapping_df_path, name_df_path):
        """
        This function constructs the mapping table from the mapping dataframe and name dataframe.
        
        Args:
        mapping_df_path: str - The path to the mapping dataframe.
        name_df_path: str - The path to the name dataframe.
        
        Returns:
        mapping_table: pd.DataFrame - The mapping table.
        """
        self.mapping_df = pd.read_csv(mapping_df_path)
        self.name_df = pd.read_csv(name_df_path)
        self.mapping_df["A_name"] = self.mapping_df["A_smiles"].map(self.name_df.set_index('Reagent SMILES')['Reagent Name'])
        self.mapping_df["B_name"] = self.mapping_df["B_smiles"].map(self.name_df.set_index('Reagent SMILES')['Reagent Name'])
        self.mapping_df["C_name"] = self.mapping_df["C_smiles"].map(self.name_df.set_index('Reagent SMILES')['Reagent Name'])
        self.mapping_df["D_name"] = self.mapping_df["D_smiles"].map(self.name_df.set_index('Reagent SMILES')['Reagent Name'])

        self.mapping_df = self.mapping_df.set_index("combined_mol_SMILES")
        
        print("Mapping table constructed.")
        
        return self.mapping_df
    
    def _load_result(self,):
        """
        This function loads the ensemble results from the result directory.
        
        Returns:
        pred_res_list: List[pd.DataFrame] - A list of dataframes containing the ensemble results.
        """
        pred_res_list = []
        for i in range(self.ensemble_num):
            pred_res_path = self.result_dir + f"final_result_{i+1}.csv"
            print("Loading ensemble result:", pred_res_path)
            pred_res = pd.read_csv(pred_res_path)
            pred_res.columns = [f"{col}_{i+1}" if col != SMI_NAME_COL else col for col in pred_res.columns]
            pred_res_list.append(pred_res)
        return pred_res_list

    def _get_smi_name(self,):
        """
        This function returns the SMILES names from the ensemble results.
        
        Returns:
        smi_name: List[str] - A list of SMILES names.
        """
        return self.pred_df[SMI_NAME_COL].tolist()

    def _preprocess_result(self, pred_res_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method combines the ensemble results into a single dataframe,
        and perform necessary preprocessing, including:
            - Merging dataframes
            - Removing duplicated columns
            - Aggregating the results by SMI_NAME
            - Reducing the results by the given reduction method
            - Reducing the ensemble results by SMI_NAME
            - Calculate the predictive variance
        
        
        Args:
            pred_res_list: List[pd.DataFrame] - A list of dataframes containing the ensemble results.
            
        Returns:
            pred_df: pd.DataFrame - The combined dataframe of the ensemble results.
            smi_reduced_pred_df: pd.DataFrame - The reduced dataframe of the ensemble results, aggregated by SMI_NAME
        """
        # merge df
        pred_df = pd.concat(pred_res_list, axis=1, join="outer")
        # remove duplicated columns
        pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]
        # reduced by SMI_NAME_COL
        smi_reduced_pred_df = pred_df.groupby(SMI_NAME_COL).agg(self.smi_reduction)
        # predictive variance
        pred_var = smi_reduced_pred_df.var(axis=1)
        # ensemble reduced by SMI_NAME_COL
        ensemble_smi_reduced_pred_df = smi_reduced_pred_df.agg(self.ensemble_reduction, axis=1)
        
        return pred_df, smi_reduced_pred_df, pred_var, ensemble_smi_reduced_pred_df

    def get_pair_plot(self, save_path: str = None, smi_reduced: bool = True):
        """
        This method generates the pair plot of the ensemble results.
        
        Args:
            save_path: str - The path to save the pair plot.
            smi_reduced: bool - Whether to use the reduced dataframe for the pair plot. Default is True.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="white")

        print("Generating pair plot...")
        if smi_reduced:
            sns.pairplot(self.smi_reduced_pred_df, diag_kind="kde", plot_kws={"s": 1.5})
        else:
            sns.pairplot(self.pred_df, diag_kind="kde", plot_kws={"s": 1.5})
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_umap_kde_plot(self, 
                          weight: Union[None, str]= None,
                          save_path: str = None):
        """
        This method generates the UMAP KDE plot of the embeddings.
        
        Args:
            weight: None | str - The weight parameter for the KDE plot.
                - None: No weight
                - "pred_var": Weight by the predictive variance
                - "pred": Weight by the predictive result
            save_path: str - The path to save the UMAP KDE plot.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="white")
        
        df = pd.DataFrame(self.dim_reduced_embeddings)
        if weight == "pred_var":
            sns.kdeplot(x=df[0], y=df[1], weights=self.pred_var.values, cmap="viridis", fill=True)
        elif weight == "pred":
            sns.kdeplot(x=df[0], y=df[1], weights=self.ensemble_smi_reduced_pred_df, cmap="viridis", fill=True)
        else:
            sns.kdeplot(x=df[0], y=df[1], cmap="viridis", fill=True)
        
        if save_path:
            plt.savefig(save_path)
    
    def count_combination(self,):
        pass
        
    def get_sankey_plot(self, smi_list: List[str], save_path: str = None):
        pass
        
    
    def get_distribution_plot(self, save_path: str = None, smi_reduced: bool = True):
        """
        
        This method generates the distribution plot of the ensemble results.
        
        Args:
            save_path: str - The path to save the distribution plot.
            smi_reduced: bool - Whether to use the reduced dataframe for the distribution plot. Default is True
        """
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="white")
        
        if self.ensemble_num > 5:
            raise ValueError("Too many ensemble models, the maximum number of ensemble models is 5.")
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        
        df = self.smi_reduced_pred_df if smi_reduced else self.pred_df

        for i in range(1, self.ensemble_num+1):
            ax = axs[(i-1)//2, (i-1)%2]
            ax.hist(df[f"pred_result_{i}"], bins=50)
            ax.set_title(f"model {i}")

        ax = axs[2, 1]

        for i in range(1, self.ensemble_num+1):
            ax.hist(df[f"pred_result_{i}"], bins=50, alpha=0.5)
        ax.set_title("gathered")
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def get_most_uncertain(self, top_k: int = 10):
        """
        This method returns the most uncertain SMILES.
        
        Args:
            top_k: int - The number of most uncertain SMILES to return.
        
        Returns:
            most_uncertain: pd.DataFrame - The most uncertain SMILES.
            index: List - The index of the most uncertain SMILES.
        """
        index = self.pred_var.nlargest(top_k).index
        most_uncertain = self.smi_reduced_pred_df.loc[index]
        return most_uncertain, index

    def get_most_certain(self, top_k: int = 10):
        """
        This method returns the most certain SMILES.
        
        Args:
            top_k: int - The number of most certain SMILES to return.
        
        Returns:
            most_certain: pd.DataFrame - The most certain SMILES.
            index: List - The index of the most certain SMILES.
        """
        index = self.pred_var.nsmallest(top_k).index
        most_certain = self.smi_reduced_pred_df.loc[index]
        return most_certain, index
    
    def get_best_performing(self, top_k: int = 10):
        """
        This method returns the top k best performing SMILES.
        
        Args:
            top_k: int - The number of top performing SMILES to return.
        
        Returns:
            best_performing: pd.DataFrame - The top k best performing SMILES.
            index: List - The index of the best performing SMILES.
        """
        index = self.ensemble_smi_reduced_pred_df.nlargest(top_k).index
        best_performing = self.smi_reduced_pred_df.loc[index]
        return best_performing, index
    
    def get_worst_performing(self, top_k: int = 10):
        """
        This method returns the top k worst performing SMILES.
        
        Args:
            top_k: int - The number of top performing SMILES to return.
            
        Returns:
            worst_performing: pd.DataFrame - The top k worst performing SMILES.
            index: List - The index of the worst performing SMILES.
        """
        index = self.ensemble_smi_reduced_pred_df.nsmallest(top_k).index
        worst_performing = self.smi_reduced_pred_df.loc[index]
        return worst_performing, index
    
    def count_component_by_smi_list(self, smi_list: List[str]):
        """
        This method counts the components of the given SMILES list.
        
        Args:
            smi_list: List[str] - A list of SMILES.
        
        Returns:
            a_count_dict: Dict - The count of A components.
            b_count_dict: Dict - The count of B components.
            c_count_dict: Dict - The count of C components.
            d_count_dict: Dict - The count of D components.
            all_count_dict: Dict - The count of all components
        """
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        all_list = []
        
        for smi in smi_list:
            entry = self.get_component_by_smi(smi)
            a_list.append(entry["A_name"])
            b_list.append(entry["B_name"])
            c_list.append(entry["C_name"])
            d_list.append(entry["D_name"])
            all_list.extend([entry["A_name"], entry["B_name"], entry["C_name"], entry["D_name"]])
        
        from collections import Counter
        a_count_dict = dict(Counter(a_list))
        b_count_dict = dict(Counter(b_list))
        c_count_dict = dict(Counter(c_list))
        d_count_dict = dict(Counter(d_list))
        all_count_dict = dict(Counter(all_list))
        
        return a_count_dict, b_count_dict, c_count_dict, d_count_dict, all_count_dict
        
    
    def get_component_by_smi(self, smi: str):
        """
        This method returns the components of the given SMILES.
        
        Args:
            smi: str - The SMILES of the molecule.
            
        Returns:
            components: Dict - The component information of the molecule.
        """
        entry = self.mapping_table.loc[smi]
        
        return {
            "A_name": entry["A_name"],
            "B_name": entry["B_name"],
            "C_name": entry["C_name"],
            "D_name": entry["D_name"],
            "A_smi": entry["A_smiles"],
            "B_smi": entry["B_smiles"],
            "C_smi": entry["C_smiles"],
            "D_smi": entry["D_smiles"],
        }
    
    def visualize_smiles(self, smi_list: List[str], save_path: str = None):
        """
        This method visualizes the given SMILES.
        
        Args:
            smi_list: List[str] - A list of SMILES to visualize.
            save_path: str - The path to save the visualization.
        """
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem import AllChem
        
        # map to each component 
        legends = []
        for smi in smi_list:
            entry = self.get_component_by_smi(smi)
            a, b, c, d = entry["A_name"], entry["B_name"], entry["C_name"], entry["D_name"]
            legends.append(f"{a} + {b} + {c} + {d}")
        mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
        
        img = Draw.MolsToGridImage(mols, 
                                   molsPerRow=5, 
                                   subImgSize=(200, 200), 
                                   legends=legends)
        if save_path:
            print("Saving the visualization to:", save_path)
            img.save(save_path)
        
    
    def visualize_top_k(self, top_k: int = 10, save_path: str = None):
        """
        This method visualizes the top k best performing SMILES.
        
        Args:
            top_k: int - The number of top performing SMILES to visualize.
            save_path: str - The path to save the visualization.
        """       
        
        top_k_sims = self.get_best_performing(top_k)[1]
        
        self.visualize_smiles(top_k_sims, save_path)
    
    def visualize_worst_k(self, top_k: int = 10, save_path: str = None):
        """
        This method visualizes the top k worst performing SMILES.
        
        Args:
            top_k: int - The number of worst performing SMILES to visualize.
            save_path: str - The path to save the visualization.
        """       
        
        worst_k_sims = self.get_worst_performing(top_k)[1]
        
        self.visualize_smiles(worst_k_sims, save_path)
        

    def _load_embedding(self,):
        """
        This method loads the embeddings from the given path, and reduce the embeddings by 
        the given reduction method.
        
        """
        with np.load(self.embedding_path) as data:
            embeddings = data[NPZ_ARR_INDEX]
        
        embeddings_df = pd.DataFrame(embeddings, index=self.full_smi_name)
        embeddings_df.index.name = SMI_NAME_COL
        if self.embedding_reduction == "first":
            embeddings_df = embeddings_df.groupby(SMI_NAME_COL).first()
        else: 
            raise ValueError("Invalid embedding reduction method.")

        return embeddings, embeddings_df
    
    def _compute_cluster(self, n_clusters=64, use_cuda=True):
        """
        This method computes the cluster of the embeddings.
        
        Args:
            n_clusters: int - The number of clusters.
            use_cuda: bool - Whether to use GPU for the clustering.
        
        Returns:
            labels: np.ndarray - The cluster labels.
        """
        if use_cuda:
            from cuml.cluster import KMeans
            # convert to float32
            embeddings = self.embeddings_df.astype(np.float32)
            kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, 
                            random_state=0).fit(self.embeddings_df)
        # attach the smi_list to kmeans.labels_
        labels = kmeans.labels_
        label_df = pd.DataFrame(labels,)
        # asisgn the self.embeddings_df.index,
        label_df.index = self.embeddings_df.index
        return label_df
    
    def get_embedding_visualization(self, save_path: str = None):
        """
        This method visualizes the embeddings.
        
        Args:
            save_path: str - The path to save the visualization.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="white")
        df = pd.DataFrame(self.dim_reduced_embeddings)
        colors = self.cluster_df[0]
        palette = sns.color_palette(cc.glasbey_hv, n_colors=len(np.unique(colors)))
        colors = [palette[i] for i in colors]
        sns.scatterplot(data=df, x=0, y=1, color=colors, alpha=0.3, s=1,)
        if save_path:
            plt.savefig(save_path)
    
    def _compute_dim_reduction(self,  reduction_method: str = "umap", use_cuda: bool = False):
        """
        This method computes the dimension reduction of the embeddings.
        
        Args:
            reduction_method: str - The reduction method, e.g. "umap", "pca", etc.
        
        """
        if not use_cuda:
            from umap import UMAP
            if reduction_method == "umap":
                reducer = UMAP()
                reduced_embeddings = reducer.fit_transform(self.embeddings_df)
            else:
                raise ValueError("Invalid reduction method.")        
        else:
            from cuml.manifold import UMAP
            print("Using GPU for dimension reduction.")
            if reduction_method == "umap":
                reducer = UMAP()
                reduced_embeddings = reducer.fit_transform(self.embeddings_df)
            else:
                raise ValueError("Invalid reduction method.")
        
        return reduced_embeddings

        
class ActiveSampler:
    def __init__(self, data_analyzer: InferenceDataAnalyzer):
        self.data_analyzer = data_analyzer
        
    def sample_best_performing(self,
                               get_k: int = 92,
                               sampling_N: int = 10000,
                               exclude: List[str] = [],
                               sampling_backend: str = "diverse"
                               ):
        """
        This method samples the best performing SMILES for active learning.
        
        Args:
            get_k: int - The number of samples to return.
            sampling_N: int - The number of samples to sample from.
        """
        most_performing, index = self.data_analyzer.get_best_performing(top_k=sampling_N)
        
        # exclude the given SMILES
        index = [idx for idx in index if idx not in exclude]
        most_performing = most_performing.loc[index]
        
        if sampling_backend == "diverse":
            sample_res = self._diverse_sampling(index, get_k)
        elif sampling_backend == "random":
            sample_res = np.random.choice(index, get_k)
        else:
            raise ValueError("Invalid sampling backend.")
        
        # get the sample_res scores 
        sample_res_scores = most_performing.loc[sample_res]
        return sample_res, sample_res_scores
    
        
    
    def sample_most_uncertain(self, 
                              get_k: int = 92, 
                              sampling_N: int = 10000, 
                              exclude: List[str] = [],
                              sampling_backend = "diverse"):
        """
        This method samples the most uncertain SMILES for active learning.
        
        Args:
            get_k: int - The number of samples to return.
            sampling_N: int - The number of samples to sample from.
            sampling_backend: str - The sampling backend, e.g. "diverse", "random"
        """
        most_uncertain, index = self.data_analyzer.get_most_uncertain(top_k=sampling_N)
        
        
        # exclude the given SMILES
        index = [idx for idx in index if idx not in exclude]
        most_uncertain = most_uncertain.loc[index]
        
        
        if sampling_backend == "diverse":
            sample_res = self._diverse_sampling(index, get_k)
        elif sampling_backend == "random":
            sample_res = np.random.choice(index, get_k)
        else:
            raise ValueError("Invalid sampling backend.")
        
        # get the sample_res scores
        sample_res_scores = most_uncertain.loc[sample_res]
        return sample_res, sample_res_scores
        
    
    
    def _diverse_sampling(self, index_list: List[str], get_k: int = 92):
        """
        This method performs diverse sampling based on the cluster information.
        
        TODO: add second field sorting.
        
        Args:
            index_list: List[str] - The list of indices.
            get_k: int - The number of samples to return.
        
        Return:
            sample_res: List[str] - The sampled indices
        """
        
        cluster_labels = self.data_analyzer.cluster_df.loc[index_list][0]
        
        # construct cross mapping
        index2cluster = dict(zip(index_list, cluster_labels))
        cluster2index = {}
        for index, cluster in index2cluster.items():
            if cluster not in cluster2index:
                cluster2index[cluster] = []
            cluster2index[cluster].append(index)
        
        # sort the cluster2index by the number of samples
        count_cluster = {cluster: len(index_list) for cluster, index_list in cluster2index.items()}
        
        # do round robin sampling
        sample_res = []
        j = 0
        while len(sample_res) < get_k:
            sorted_clusters = sorted(count_cluster.items(), key=lambda x: x[1], reverse=True)
            cluster = sorted_clusters[j % len(sorted_clusters)][0]
            cluster_indices = cluster2index[cluster]
            if len(cluster_indices) == 0:
                j += 1
                continue
            selected_idx = np.random.choice(cluster_indices, 1)[0]
            # remove selected index from cluster2idx_mapping
            cluster2index[cluster].remove(selected_idx)
            count_cluster[cluster] -= 1
            sample_res.append(selected_idx)
        
        return sample_res
        
        
        
        
        
        
if __name__ == "__main__":
    result_dir = "/home/sdl/SDL-LNP/model/serverless/test/"
    inference_data_analyer = InferenceDataAnalyzer(result_dir=result_dir, 
                                        embedding_path="/home/sdl/SDL-LNP/model/evaluation/notebooks/test_result_representation.npz",
                                        mapping_df_path="/home/sdl/SDL-LNP/model/data_process/220k_library.csv",
                                        name_df_path="/home/sdl/SDL-LNP/mapping_table/General_mapping_sampler.csv",
                                        use_cuda=False)
    inference_data_analyer.get_umap_kde_plot(weight=None, save_path="umap_kde.png")
    active_learner = ActiveSampler(inference_data_analyer)
    print(active_learner.sample_most_uncertain())

