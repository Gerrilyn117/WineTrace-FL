import random
import numpy as np
import os
import pandas as pd
import torch
import openml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from scipy.special import softmax
import xgboost as xgb
import time
from torch.optim import Adam

# Global parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!!")
RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments')
os.makedirs(RESULT_DIR, exist_ok=True)
CONTRASTIVE_LEARNING_MAX_EPOCHS = 250
SUPERVISED_LEARNING_MAX_EPOCHS = 500
CLS_CORR_REFRESH_SAMPLER_PERIOD = 10
LR = 0.001
FRACTION_LABELED = 1
CORRUPTION_RATE = 0.3
BATCH_SIZE = 512
SEEDS = [42]
assert len(SEEDS) == len(set(SEEDS))
ALL_METHODS = ['no_pretrain',
               'orc_corr-rand_feats']
P_VAL_SIGNIFICANCE = 0.05
CORRELATED_FEATURES_RANDOMIZE_SAMPLING = True
CORRELATED_FEATURES_RANDOMIZE_SAMPLING_TEMPERATURE = 0.1
# Result processing metric
METRIC = "accuracy"
XGB_FEATURECORR_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "eta": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "enable_categorical": True,
    "tree_method": "hist"
}

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_openml_list(DIDS):
    datasets = []
    datasets_list = openml.datasets.list_datasets(DIDS, output_format='dataframe')

    for ds in datasets_list.index:
        entry = datasets_list.loc[ds]

        print('Loading', entry['name'], entry.did, '..')
        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression tasks are not supported yet")
            exit(1)
        else:
            dataset = openml.datasets.get_dataset(int(entry.did))
            # since under SCARF corruption, the replacement by sampling happens before one-hot encoding, load the
            # data in its original form
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )

            assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)

            order = np.arange(y.shape[0])
            np.random.shuffle(order)
            X, y = X.iloc[order], y.iloc[order]

            assert X is not None

        datasets += [[entry['name'],
                      entry.did,
                      int(entry['NumberOfClasses']),
                      np.sum(categorical_indicator),
                      len(X.columns),
                      X,
                      y]]

    return datasets


def preprocess_datasets(train_data, test_data, normalize_numerical_features):
    assert isinstance(train_data, pd.DataFrame) and \
           isinstance(test_data, pd.DataFrame)
    assert np.all(train_data.columns == test_data.columns)
    features_dropped = []
    for col in train_data.columns:
        # drop columns with all null values or with a constant value on training data
        if train_data[col].isnull().all() or train_data[col].nunique() == 1:
            train_data.drop(columns=col, inplace=True)
            test_data.drop(columns=col, inplace=True)
            features_dropped.append(col)
            continue
        # fill the missing values
        if train_data[col].isnull().any() or test_data[col].isnull().any():
            # for categorical features, fill with the mode in the training data
            if train_data[col].dtype.name == "category":
                val_fill = train_data[col].mode(dropna=True)[0]
            # for numerical features, fill with the mean of the training data
            else:
                val_fill = train_data[col].mean(skipna=True)
            train_data[col].fillna(val_fill, inplace=True)
            test_data[col].fillna(val_fill, inplace=True)

    if normalize_numerical_features:
        # z-score transform numerical values
        scaler = StandardScaler()
        non_categorical_cols = train_data.select_dtypes(exclude='category').columns
        if len(non_categorical_cols) == 0:
            print("No numerical features! Skipping z-score normalization for numerical features.")
        else:
            train_data[non_categorical_cols] = scaler.fit_transform(train_data[non_categorical_cols])
            test_data[non_categorical_cols] = scaler.transform(test_data[non_categorical_cols])
    print(
        f"Data preprocessing complete! {len(features_dropped)} features dropped: {features_dropped}. {'Numerical features normalized.' if normalize_numerical_features else ''}")
    # retain the pandas dataframe format for later one-hot encoder
    return train_data, test_data

def fit_one_hot_encoder(one_hot_encoder_raw, train_data):
    categorical_cols = train_data.select_dtypes(include='category').columns
    one_hot_encoder = make_column_transformer((one_hot_encoder_raw, categorical_cols),
                                              remainder='passthrough')
    one_hot_encoder.fit(train_data)
    return one_hot_encoder


def initialize_adam_optimizer(model):
    return Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=LR)

def evaluate_and_save_results(model,
                              test_data,
                              test_targets,
                              one_hot_encoder,
                              save_name="default",
                              class_names=None,
                              device=DEVICE):
    model.eval()
    with torch.no_grad():
        # one-hot
        inputs = one_hot_encoder.transform(pd.DataFrame(data=test_data))
        inputs = torch.tensor(inputs.astype(float), dtype=torch.float32).to(device)
        test_targets = torch.tensor(test_targets.astype(int), dtype=torch.int64).to(device)

        pred_logits = model.module.get_classification_prediction_logits(inputs)
        y_pred = pred_logits.argmax(dim=1).cpu().numpy()
        y_true = test_targets.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        macro_precision = report_dict["macro avg"]["precision"]
        macro_recall = report_dict["macro avg"]["recall"]
        macro_f1 = report_dict["macro avg"]["f1-score"]

        print(f"\nAccuracy: {acc:.4f}")
        print(f"MCC: {mcc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=4))

        summary_df = pd.DataFrame([{
            "MCC": mcc,
            "ACC": acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1-score": macro_f1
        }])
        summary_path = fr"liquor/result/classification_summary_{save_name}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nMetrics saved to: {summary_path}")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else np.unique(y_true),
                    yticklabels=class_names if class_names else np.unique(y_true))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - {save_name}')
        plt.tight_layout()

        # 保存图像
        fig_path = fr"liquor/result/confusion_matrix_{save_name}.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Confusion matrix image saved to: {fig_path}")

        return summary_df

def save_train_metrics_separately(dataset_name, loss_histories, acc_histories):
    output_dir = rf"liquor/result/{dataset_name}/train_loss&acc"
    os.makedirs(output_dir, exist_ok=True)

    for method in loss_histories:
        losses = loss_histories[method]
        accs = acc_histories[method]
        records = []

        for epoch in range(len(losses)):
            records.append({
                "epoch": epoch,
                "train_loss": losses[epoch],
                "train_acc": accs[epoch]
            })

        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, f"{method}_train_metrics.csv")
        df.to_csv(csv_path, index=False)

def save_fold_histories(fold_idx, dataset, contrastive_loss_histories, supervised_loss_histories,
                        supervised_accuracy_histories=None, save_root=None):
    """
    Save training loss/accuracy records for a fold.

    Args:
    - fold_idx: current fold index (starting from 0)
    - contrastive_loss_histories: dict, contrastive loss records for each method
    - supervised_loss_histories: dict, supervised loss records for each method
    - supervised_accuracy_histories: dict, supervised accuracy records for each method, optional
    - save_root: str, root directory to save, default is "loss_logs" in current working directory
    """
    save_dataset = dataset
    if save_root is None:
        save_root = os.path.join(os.getcwd(), fr"loss_logs/{save_dataset}")
    os.makedirs(save_root, exist_ok=True)

    fold_dir = os.path.join(save_root, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for method, losses in contrastive_loss_histories.items():
        save_path = os.path.join(fold_dir, f"{method}_contrastive_loss.csv")
        df = pd.DataFrame({'epoch': list(range(1, len(losses) + 1)), 'loss': losses})
        df.to_csv(save_path, index=False)

    for method, losses in supervised_loss_histories.items():
        save_path = os.path.join(fold_dir, f"{method}_supervised_loss.csv")
        df = pd.DataFrame({'epoch': list(range(1, len(losses) + 1)), 'loss': losses})
        df.to_csv(save_path, index=False)

    if supervised_accuracy_histories is not None:
        for method, accs in supervised_accuracy_histories.items():
            save_path = os.path.join(fold_dir, f"{method}_supervised_acc.csv")
            df = pd.DataFrame({'epoch': list(range(1, len(accs) + 1)), 'accuracy': accs})
            df.to_csv(save_path, index=False)

    print(f"Fold {fold_idx + 1} training records saved to {fold_dir}")

def save_fold_results(train_metrics_df, test_metrics_df, fold_idx, methods, dataset):

    import os
    save_dir = f"results/{dataset}/fold_{fold_idx}"
    os.makedirs(save_dir, exist_ok=True)

    merged_df = pd.concat(
        [train_metrics_df, test_metrics_df],
        axis=1,
        keys=["Train", "Test"]
    )
    merged_df.to_csv(f"{save_dir}/{methods}_metrics.csv")
    return merged_df

def aggregate_five_folds(methods, dataset, n_folds=5):
    out_file = f"results/{dataset}/{methods}_5fold_mean_std.csv"
    dfs = []
    for k in range(0, n_folds):
        df = pd.read_csv(f"results/{dataset}/fold_{k}/{methods}_metrics.csv", index_col=0, header=[0, 1])
        dfs.append(df)

    arr = np.stack([df.values for df in dfs], axis=0)

    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    formatted = np.empty_like(means, dtype=object)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            formatted[i, j] = f"{means[i, j]:.4f} ± {stds[i, j]:.4f}"

    final_df = pd.DataFrame(formatted, index=dfs[0].index, columns=dfs[0].columns)
    final_df.to_csv(out_file)
    return final_df

def compute_metrics(y_true, y_pred, y_prob):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_classes = len(np.unique(y_true))

    acc_total = accuracy_score(y_true, y_pred)
    precision_total = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_total = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc_total = matthews_corrcoef(y_true, y_pred)
    try:
        auc_total = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc_total = np.nan

    acc_per_class, auc_per_class, mcc_per_class = [], [], []
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    for c in range(n_classes):
        mask = (y_true == c)
        if mask.sum() > 0:
            acc_c = accuracy_score(y_true[mask], y_pred[mask])
        else:
            acc_c = np.nan
        acc_per_class.append(acc_c)

        try:
            auc_c = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
        except Exception:
            auc_c = np.nan
        auc_per_class.append(auc_c)

        mcc_c = matthews_corrcoef((y_true == c).astype(int), (y_pred == c).astype(int))
        mcc_per_class.append(mcc_c)

    df = pd.DataFrame({
        "ACC": acc_per_class + [acc_total],
        "P": list(precision_per_class) + [precision_total],
        "R": list(recall_per_class) + [recall_total],
        "AUC": auc_per_class + [auc_total],
        "MCC": mcc_per_class + [mcc_total]
    }, index=[str(i + 1) for i in range(n_classes)] + ["total"])

    return df, acc_total

def save_train_test_metrics(dataset_did, method, train_results, test_results):
    train_df, train_acc_total = compute_metrics(train_results["y_true"], train_results["y_pred"],
                                                train_results["y_prob"])
    test_df, test_acc_total = compute_metrics(test_results["y_true"], test_results["y_pred"], test_results["y_prob"])

    final_df = pd.concat([train_df.add_prefix("Train_"), test_df.add_prefix("Test_")], axis=1)

    save_dir = os.path.join("results", str(dataset_did))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{method}_5fold.csv")

    final_df.to_csv(save_path, encoding="utf-8-sig")
    print(f"✅ Saved to: {save_path}")
    return train_acc_total, test_acc_total