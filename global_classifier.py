import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
from sklearn import datasets, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import (matthews_corrcoef, ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm
from collections import defaultdict
from model import Neural_Net
from dataset_samplers import RandomCorruptSampler, ClassCorruptSampler, SupervisedSampler
from corruption_mask_generators import RandomMaskGenerator, CorrelationMaskGenerator

from training import train_contrastive_loss, train_classification, train_classification_all_metrics
from utils import *

import warnings

warnings.filterwarnings('ignore')
print("Disabled warnings!")

print(f"Using DEVICE: {DEVICE}")

if __name__ == "__main__":
    k = 5
    dataset_dids = ["R=6D"]
    for dataset_did in dataset_dids:
        test_results_all_folds = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})
        train_results_all_folds = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})
        for kfold in range(k):
            dataset_name = f"liquor3{kfold + 1}"  # "liquor31","liquor32","liquor33","liquor34","liquor35"
            print(f"Reading fold {kfold + 1} file for {dataset_did}")
            n_feats_before_processing = 52
            n_cat_feats_before_processing = 3

            if dataset_did == 'real':
                train_data = rf"/root/autodl-tmp/Tabular-Class-Conditioned-SSL-main/liquor/data/5fold/{dataset_name}/{dataset_name}_train.csv"
            else:
                train_data = rf"/root/autodl-tmp/Tabular-Class-Conditioned-SSL-main/liquor/data/5fold0823/{dataset_name}/{dataset_did}/{dataset_name}_combined_syn_data.csv"
            test_data = rf"/root/autodl-tmp/Tabular-Class-Conditioned-SSL-main/liquor/data/5fold/{dataset_name}/{dataset_name}_test.csv"
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            csv_filename_train = os.path.splitext(os.path.basename(train_data))[0]
            csv_filename_test = os.path.splitext(os.path.basename(test_data))[0]
            print(f'Starting training for fold {kfold + 1} of {dataset_did}, training set: {csv_filename_train}, test set: {csv_filename_test}')

            accuracies, aurocs = {}, {}
            for key in ALL_METHODS:
                accuracies[key], aurocs[key] = [], []
            seed = 1234
            fix_seed(seed)

            train_data = train_df.drop(columns=["y"]).values
            train_targets = train_df["y"].values

            test_data = test_df.drop(columns=["y"]).values
            test_targets = test_df["y"].values

            label_encoder_target = preprocessing.LabelEncoder()
            train_targets = label_encoder_target.fit_transform(train_targets)
            test_targets = label_encoder_target.transform(test_targets)

            print(f"Class distributions:")
            unique, counts = np.unique(train_targets, return_counts=True)
            print(f"Training: cls: {unique}; counts: {counts}")
            unique, counts = np.unique(test_targets, return_counts=True)
            print(f"Testing: cls: {unique}; counts: {counts}")

            n_classes = len(np.unique(train_targets))

            train_data, test_data = preprocess_datasets(pd.DataFrame(train_data), pd.DataFrame(test_data),
                                                                                  normalize_numerical_features=True)
            one_hot_encoder = fit_one_hot_encoder(preprocessing.OneHotEncoder(handle_unknown='ignore', \
                                                                              drop='if_binary', \
                                                                              sparse_output=False), \
                                                  train_data)

            n_train_samples_labeled = int(len(train_data) * FRACTION_LABELED)
            idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
            mask_train_labeled = np.zeros(len(train_data), dtype=bool)
            mask_train_labeled[idxes_tmp] = True
            supervised_sampler = SupervisedSampler(data=train_data[mask_train_labeled],
                                                   target=train_targets[mask_train_labeled])

            feat_impt, feat_impt_range = compute_feature_mutual_influences(train_data)

            # prepare models
            models, contrastive_loss_histories, supervised_loss_histories, supervised_accuracy_histories, supervised_all_metrics_histories, supervised_all_epoch_histories = {}, {}, {}, {}, {}, {}

            for method in ALL_METHODS:
                models[method] = nn.DataParallel(Neural_Net(
                    input_dim=one_hot_encoder.transform(train_data).shape[1],  # model expect one-hot encoded input
                    emb_dim=256,
                    output_dim=n_classes
                )).to(DEVICE)

            # Firstly, train the supervised learning model on the labeled subset
            # freeze the supervised learning encoder as initialized
            models['no_pretrain'].module.freeze_encoder()
            print("Supervised training for no_pretrain...")
            train_losses, train_accuracies, all_metrics, all_epoch_results = train_classification_all_metrics(models['no_pretrain'], supervised_sampler, one_hot_encoder)
            supervised_loss_histories['no_pretrain'] = train_losses
            supervised_accuracy_histories['no_pretrain'] = train_accuracies
            supervised_all_metrics_histories['no_pretrain'] = all_metrics
            supervised_all_epoch_histories['no_pretrain'] = all_epoch_results

            ############# Prepare data samplers for corruption############
            contrastive_samplers = {}
            # Random Sampling: Ignore class information in original corruption
            contrastive_samplers['rand_corr'] = RandomCorruptSampler(train_data)
            # Oracle Class Sampling: Use oracle info on training labels
            contrastive_samplers['orc_corr'] = ClassCorruptSampler(train_data, train_targets)
            # Predicted Class Sampling: Use supervised model to obtain pseudo labels at the beginning
            bootstrapped_train_targets = get_bootstrapped_targets( \
                train_data, train_targets, models['no_pretrain'], mask_train_labeled, one_hot_encoder)
            contrastive_samplers['cls_corr'] = ClassCorruptSampler(train_data, bootstrapped_train_targets)

            ################ Prepare feature selections for masking #############
            # prepare mask generator
            mask_generators = {}
            mask_generators['rand_feats'] = RandomMaskGenerator(train_data.shape[1])
            mask_generators['leastRela_feats'] = CorrelationMaskGenerator(train_data.shape[1], high_correlation=False)
            mask_generators['mostRela_feats'] = CorrelationMaskGenerator(train_data.shape[1], high_correlation=True)
            mask_generators['leastRela_feats'].initialize_feature_importances(feat_impt)
            mask_generators['mostRela_feats'].initialize_feature_importances(feat_impt)

            for method in ALL_METHODS:
                if method == "no_pretrain":
                    continue
                assert '-' in method
                corrupt_method, corrupt_loc = method.split('-')
                # Contrastive training
                train_losses = train_contrastive_loss(models[method],
                                                      method,
                                                      contrastive_samplers[corrupt_method],
                                                      supervised_sampler,
                                                      mask_generators[corrupt_loc],
                                                      mask_train_labeled,
                                                      one_hot_encoder)
                contrastive_loss_histories[method] = train_losses

                # fine tune the pre-trained models on the down-stream supervised learning task
                models[method].module.freeze_encoder()
                print(f"Supervised fine-tuning for {method}...")
                train_losses, train_accuracies, all_metrics, all_epoch_results = train_classification_all_metrics(models[method], supervised_sampler, one_hot_encoder)
                supervised_loss_histories[method] = train_losses
                supervised_accuracy_histories[method] = train_accuracies
                supervised_all_metrics_histories[method] = all_metrics
                supervised_all_epoch_histories[method] = all_epoch_results

            # save_train_metrics_separately(dataset_name, supervised_loss_histories, supervised_accuracy_histories)
            # Evaluation on prediction accuracies and aucs
            for method in ALL_METHODS:
                models[method].module.eval()
                with torch.no_grad():
                    test_logits = models[method].module.get_classification_prediction_logits(
                        torch.tensor(one_hot_encoder.transform(test_data), dtype=torch.float32).to(DEVICE)
                    )
                    test_probs = test_logits.softmax(dim=1).cpu().numpy()
                    test_preds = np.argmax(test_probs, axis=1)

                    n_classes = len(np.unique(test_targets))
                    y_true = np.array(test_targets)

                    acc_total = accuracy_score(y_true, test_preds)
                    precision_total = precision_score(y_true, test_preds, average="macro", zero_division=0)
                    recall_total = recall_score(y_true, test_preds, average="macro", zero_division=0)
                    mcc_total = matthews_corrcoef(y_true, test_preds)
                    try:
                        if n_classes == 2:
                            auc_total = roc_auc_score(y_true, test_probs[:, 1])
                        else:
                            auc_total = roc_auc_score(y_true, test_probs, multi_class="ovr")
                    except ValueError:
                        auc_total = np.nan

                    acc_per_class = []
                    auc_per_class = []
                    mcc_per_class = []
                    precision_per_class = precision_score(y_true, test_preds, average=None, zero_division=0)
                    recall_per_class = recall_score(y_true, test_preds, average=None, zero_division=0)

                    for c in range(n_classes):
                        mask = (y_true == c)
                        if mask.sum() > 0:
                            acc_c = accuracy_score(y_true[mask], test_preds[mask])
                        else:
                            acc_c = np.nan
                        acc_per_class.append(acc_c)

                        try:
                            auc_c = roc_auc_score((y_true == c).astype(int), test_probs[:, c])
                        except ValueError:
                            auc_c = np.nan
                        auc_per_class.append(auc_c)

                        mcc_c = matthews_corrcoef((y_true == c).astype(int), (test_preds == c).astype(int))
                        mcc_per_class.append(mcc_c)

                    test_metrics_df = pd.DataFrame({
                        "ACC": acc_per_class + [acc_total],
                        "AUC": auc_per_class + [auc_total],
                        "MCC": mcc_per_class + [mcc_total],
                        "Precision": list(precision_per_class) + [precision_total],
                        "Recall": list(recall_per_class) + [recall_total],
                    }, index=[f"class_{i}" for i in range(n_classes)] + ["total"])

                    train_metrics_df = supervised_all_metrics_histories[method][-1]

                    save_fold_results(train_metrics_df, test_metrics_df, fold_idx=kfold, methods = method, dataset = dataset_did)

                    test_results_all_folds[method]["y_true"].extend(test_targets)
                    test_results_all_folds[method]["y_pred"].extend(test_preds)
                    test_results_all_folds[method]["y_prob"].extend(test_probs)

                    last_epoch_result = all_epoch_results[-1]
                    train_results_all_folds[method]["y_true"].extend(last_epoch_result["targets"])
                    train_results_all_folds[method]["y_pred"].extend(last_epoch_result["preds"])
                    train_results_all_folds[method]["y_prob"].extend(last_epoch_result["pred_logits"])

            print(f"<<<<<<<<<<<<<<<{dataset_name} finished!>>>>>>>>>>>>>>")
        print("Main function finished!")
        for method in ALL_METHODS:
            print(f"\nComprehensive 5-fold evaluation results for {dataset_did}_{method}:")

            train_acc_total, test_acc_total= save_train_test_metrics(dataset_did, method,
                                                                    train_results_all_folds[method],
                                                                    test_results_all_folds[method])

            test_y_true = np.array(test_results_all_folds[method]["y_true"])
            test_y_pred = np.array(test_results_all_folds[method]["y_pred"])
            test_y_prob = np.array(test_results_all_folds[method]["y_prob"])

            train_y_true = np.array(train_results_all_folds[method]["y_true"])
            train_y_pred = np.array(train_results_all_folds[method]["y_pred"])
            train_y_prob = np.array(train_results_all_folds[method]["y_prob"])

            pic_dir = fr"/root/autodl-tmp/Tabular-Class-Conditioned-SSL-main/liquor/result/{dataset_did}/pic"
            os.makedirs(pic_dir, exist_ok=True)
            classes = ["HL", "CJ", "JD"]
            cm_test = confusion_matrix(test_y_true, test_y_pred)
            cm_test_df = pd.DataFrame(cm_test, columns=classes, index=classes)
            csv_path_test = os.path.join(pic_dir, f"confusion_matrix_test_{dataset_did}_{method}.csv")
            cm_test_df.to_csv(csv_path_test)
            print(f"✅ Test set confusion matrix saved to: {csv_path_test}")

            cm_train = confusion_matrix(train_y_true, train_y_pred)
            cm_train_df = pd.DataFrame(cm_train, columns=classes, index=classes)
            csv_path_train = os.path.join(pic_dir, f"confusion_matrix_train_{dataset_did}_{method}.csv")
            cm_train_df.to_csv(csv_path_train)
            print(f"✅ Train set confusion matrix saved to: {csv_path_train}")

            print(f"train acc: {train_acc_total}\ntest acc: {test_acc_total}")