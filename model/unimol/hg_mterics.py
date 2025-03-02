import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import rdkit
from rdkit.Chem import Draw

def relaxed_spearman_correlation(preds, targets, relax_ratio=0.05):
    """
    Compute the relaxed spearman correlation. The relax ratio tells the amount of difference allowed. A delta threshold will be computed by the relax ratio times the dynamica range of the target values. For each pair of values, if the difference is smaller than the delta threshold, will make the difference to be zero.

    Args:
        preds (np.ndarray): The predicted values.
        target (np.ndarray): The target values.
        relax_ratio (float): The relax ratio.
    """

    assert len(preds) == len(
        targets
    ), "The length of preds and target should be the same."
    n = len(preds)

    # Use rankdata to correctly handle ties
    x_rank = rankdata(preds)
    y_rank = rankdata(targets)
    delta = relax_ratio * n

    # Calculate the difference in ranks
    d = x_rank - y_rank
    d = np.where(np.abs(d) <= delta, 0, d)

    # Calculate the sum of the squared differences
    d_squared_sum = np.sum(d**2)

    # Calculate the Spearman correlation coefficient
    correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return correlation

def compile_test_metrics(hf_test_preds, hash2smi, select_topk_conformer:int= 11):
    target = hf_test_preds.label_ids[0].squeeze()
    prediction = hf_test_preds.predictions[0].squeeze()
    smi_name = hf_test_preds.label_ids[1].squeeze()
    
    df = pd.DataFrame(
        {
            "smi_name": smi_name,
            "target": target,
            "prediction": prediction,
        }
    )
    
    df["smi_name"] = df["smi_name"].apply(lambda x: hash2smi[int(x)])
    
    # df = df.groupby("smi_name").mean().reset_index()
    df = df.groupby("smi_name").head(select_topk_conformer).groupby("smi_name").mean().reset_index()
    
    log2wandb(df["target"].values, df["prediction"].values, df["smi_name"].values)



def compute_metrics(smi_string=None, 
                    drop_extrema=False,
                    keep_topk_conformer=None,):
    def compute_metrics_inner(pred):
        target = pred.label_ids[0].squeeze()
        prediction = pred.predictions[0].squeeze()
        smi_name = pred.label_ids[1].squeeze()

        df = pd.DataFrame(
            {
                "smi_name": smi_name,
                "target": target,
                "prediction": prediction,
            }
        )

        if not drop_extrema:
            df = df.groupby("smi_name").mean().reset_index()
        else:
            # Drop the highest and lowest entries for same smi_name
            # assume each smi_name has 10 entries
            df = df.groupby("smi_name").apply(
                lambda x: x[(x["prediction"] > x["prediction"].quantile(0.1)) & (x["prediction"] < x["prediction"].quantile(0.9))].mean()
            ).reset_index(drop=True)
        
        if keep_topk_conformer is not None:
            df = df.groupby("smi_name").head(keep_topk_conformer).groupby("smi_name").mean().reset_index()
                        
        top_25_target = df["target"].quantile(0.75)
        top25_target_df = df[df["target"] >= top_25_target]
        
        top_25_prediction = df["prediction"].quantile(0.75)
        top25_prediction_df = df[df["prediction"] >= top_25_prediction]
        
        top25_target_set = set(top25_target_df["smi_name"])
        top25_prediction_set = set(top25_prediction_df["smi_name"])
        
        shared_top25 = top25_target_set.intersection(top25_prediction_set)
        top25_agreement_percentage = len(shared_top25) / len(top25_target_set)
        
        non_zero_df = df[df["target"] > 0]
        
        
        return {
            "pearson": pearsonr(df["target"].values, df["prediction"].values)[0],
            "non_zero_pearson": pearsonr(non_zero_df["target"].values, non_zero_df["prediction"].values)[0],
            "spearman": spearmanr(df["target"].values, df["prediction"].values)[0],
            "non_zero_spearman": spearmanr(non_zero_df["target"].values, non_zero_df["prediction"].values)[0],
            "relaxed_spearman": relaxed_spearman_correlation(
                df["target"].values, df["prediction"].values
            ),
            "non_zero_relaxed_spearman": relaxed_spearman_correlation(
                non_zero_df["target"].values, non_zero_df["prediction"].values
            ),
            "mse": np.mean((df["target"].values - df["prediction"].values) ** 2),
            "non_zero_mse": np.mean((non_zero_df["target"].values - non_zero_df["prediction"].values) ** 2),
            "top25_pearson": pearsonr(top25_target_df["target"].values, top25_target_df["prediction"].values)[0],
            "top25_spearman": spearmanr(top25_target_df["target"].values, top25_target_df["prediction"].values)[0],
            "top25_relaxed_spearman": relaxed_spearman_correlation(
                top25_target_df["target"].values, top25_target_df["prediction"].values
            ),
            "top25_mse": np.mean((top25_target_df["target"].values - top25_target_df["prediction"].values) ** 2),
            "top25_agreement_percentage": top25_agreement_percentage,
        }
    return compute_metrics_inner
        


def plot_smi_name(smi_name: str):
    # Plot the chemical structure of smi_name
    image = Draw.MolToImage(rdkit.Chem.MolFromSmiles(smi_name))
    return wandb.Image(image)

    


def log2wandb(target, pred, smi_name):
    residuals = target - pred
    data = [[target[i], pred[i]] for i in range(len(target))]
    table = wandb.Table(data=data, columns=["target", "prediction"])

    error = np.abs(residuals)
    
    
    # top10 error samples
    top10_idx = np.argsort(error)[::-1][:10]
    if smi_name is not None:
        top10_error_data = [[smi_name[i], plot_smi_name(smi_name[i]), target[i], pred[i], residuals[i]] for i in top10_idx]
        col_name  = ["smi_name", "chemical_structure", "target", "prediction", "residual"]
    else:
        top10_error_data = [[target[i], pred[i], residuals[i]] for i in top10_idx]
        col_name = ["target", "prediction", "residual"]
    
    # top10 best samples
    top10_best_idx = np.argsort(error)[:10]
    if smi_name is not None:
        top10_best_data = [[smi_name[i], plot_smi_name(smi_name[i]), target[i], pred[i], residuals[i]] for i in top10_best_idx]
    else: 
        top10_best_data = [[target[i], pred[i], residuals[i]] for i in top10_best_idx]

    
    pred_dist = wandb.Histogram(
        pred,
    )
    
    
    return_data={
            "test_scatter": wandb.plot.scatter(
                table,
                x="target",
                y="prediction",
                title="Target vs Prediction",
            ),
            "test_residuals": wandb.Histogram(
                residuals,
            ),
            "test_top10_error": wandb.Table(data=top10_error_data, 
                                       columns=col_name),
            "test_top10_best": wandb.Table(data=top10_best_data, 
                                      columns=col_name),
            "test_pred_dist": pred_dist,
        }
    wandb.log(return_data)
    return {}

            

class NestedKFold:
    
    """This method creates rolling testing set,
    where the testing set is the last n_splits folds.
    
    Remaining folds are used for training and validation.
    """
    
    def __init__(self, n_splits=5, random_state=42, shuffle=False, train_size=0.7):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.train_size = train_size
        
    def split(self, data):
        """Return training, validation and testing
        """
        
        n = len(data)
        idx = np.arange(n)
        if self.shuffle:
            np.random.seed(self.random_state)
            idx = np.random.permutation(idx)
        
        for i in range(self.n_splits):
            start = i * n // self.n_splits
            end = (i + 1) * n // self.n_splits
            train_idx_and_valid = np.concatenate([idx[:start], idx[end:]])
            
            train_idx = train_idx_and_valid[:int(self.train_size * len(train_idx_and_valid))]
            valid_idx = train_idx_and_valid[int(self.train_size * len(train_idx_and_valid)):]
            
            test_idx = idx[start:end]
            yield train_idx, valid_idx, test_idx

class ProgressionKFold:
    
    """This method creates rolling testing set,
    where the testing set is the last n_splits folds.
    
    Remaining data that are ordered before the testing set are used for training and validation.
    """
    
    def __init__(self, n_splits=5, random_state=42, shuffle=False, train_size=0.7):
        self.n_splits = n_splits + 1
        self.random_state = random_state
        self.shuffle = shuffle
        self.train_size = train_size
        
    def split(self, data):
        """Return training, validation and testing
        """
        
        n = len(data)
        idx = np.arange(n)
        if self.shuffle:
            np.random.seed(self.random_state)
            idx = np.random.permutation(idx)
        
        for i in range(1, self.n_splits):
            start = i * n // self.n_splits
            end = (i + 1) * n // self.n_splits
            train_idx_and_valid = idx[:start]
            
            train_idx = train_idx_and_valid[:int(self.train_size * len(train_idx_and_valid))]
            valid_idx = train_idx_and_valid[int(self.train_size * len(train_idx_and_valid)):]
            
            test_idx = idx[start:end]
            yield train_idx, valid_idx, test_idx