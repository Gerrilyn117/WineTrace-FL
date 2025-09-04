import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataset_samplers import ClassCorruptSampler
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
def _train_contrastive_loss_oneEpoch(model, 
                                     data_sampler, 
                                     mask_generator, 
                                     optimizer,
                                     one_hot_encoder):
    model.train()
    epoch_loss = 0
    for _ in range(data_sampler.n_batches):
        anchors, random_samples = data_sampler.sample_batch()
        # firstly, corrupt on the original pandas dataframe
        corruption_masks = mask_generator.get_masks(np.shape(anchors)[0])
        assert np.shape(anchors) == np.shape(corruption_masks)
        anchors_corrupted = np.where(corruption_masks, random_samples, anchors)
        # after corruption, do one-hot encoding
        anchors, anchors_corrupted = one_hot_encoder.transform(pd.DataFrame(data=anchors,columns=data_sampler.columns)), \
                                        one_hot_encoder.transform(pd.DataFrame(data=anchors_corrupted,columns=data_sampler.columns))

        anchors, anchors_corrupted = torch.tensor(anchors.astype(float), dtype=torch.float32).to(DEVICE), \
                                        torch.tensor(anchors_corrupted.astype(float), dtype=torch.float32).to(DEVICE)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_final_anchors = model.module.get_final_embedding(anchors)
        emb_final_corrupted = model.module.get_final_embedding(anchors_corrupted)

        # compute loss
        loss = model.module.contrastive_loss(emb_final_anchors, emb_final_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / data_sampler.n_batches


def train_contrastive_loss(model, 
                           method_key, 
                           contrastive_sampler, 
                           supervised_sampler, 
                           mask_generator, 
                           mask_train_labeled, 
                           one_hot_encoder):
    print(f"Contrastive learning for {method_key}....")
    train_losses = []
    optimizer = initialize_adam_optimizer(model)
    
    for i in tqdm(range(1, CONTRASTIVE_LEARNING_MAX_EPOCHS+1)):
        epoch_loss = _train_contrastive_loss_oneEpoch(model, 
                                                      contrastive_sampler, 
                                                      mask_generator, 
                                                      optimizer, 
                                                      one_hot_encoder)
        train_losses.append(epoch_loss)

    return train_losses

def train_classification(model, supervised_sampler, one_hot_encoder):
    train_losses = []
    train_accuracies = []
    optimizer = initialize_adam_optimizer(model)
    model.module.initialize_classification_head()

    for _ in range(SUPERVISED_LEARNING_MAX_EPOCHS):
        model.module.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        for _ in range(supervised_sampler.n_batches):
            inputs, targets = supervised_sampler.sample_batch()
            inputs = one_hot_encoder.transform(pd.DataFrame(data=inputs, columns=supervised_sampler.columns))
            inputs = torch.tensor(inputs.astype(float), dtype=torch.float32).to(DEVICE)
            # seemingly int64 is often used as the type for indices
            targets = torch.tensor(targets.astype(int), dtype=torch.int64).to(DEVICE)

            # reset gradients
            optimizer.zero_grad()

            # get classification predictions
            pred_logits = model.module.get_classification_prediction_logits(inputs)

            # compute loss
            loss = model.module.classification_loss(pred_logits, targets)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(pred_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        train_losses.append(epoch_loss / supervised_sampler.n_batches)
        acc = accuracy_score(all_targets, all_preds)
        train_accuracies.append(acc)
    return train_losses, train_accuracies

def train_classification_all_metrics(model, supervised_sampler, one_hot_encoder):
    train_losses = []
    train_accuracies = []
    all_metrics = []
    all_epoch_results = []
    optimizer = initialize_adam_optimizer(model)
    model.module.initialize_classification_head()

    for _ in range(SUPERVISED_LEARNING_MAX_EPOCHS):
        model.module.train()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        for _ in range(supervised_sampler.n_batches):
            inputs, targets = supervised_sampler.sample_batch()

            inputs = one_hot_encoder.transform(pd.DataFrame(data=inputs, columns=supervised_sampler.columns))
            inputs = torch.tensor(inputs.astype(float), dtype=torch.float32).to(DEVICE)
            # seemingly int64 is often used as the type for indices
            targets = torch.tensor(targets.astype(int), dtype=torch.int64).to(DEVICE)

            # reset gradients
            optimizer.zero_grad()

            # get classification predictions
            pred_logits = model.module.get_classification_prediction_logits(inputs)

            # compute loss
            loss = model.module.classification_loss(pred_logits, targets)
            loss.backward()

            # update model weights
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(pred_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.extend(pred_logits.detach().cpu().numpy())

        train_losses.append(epoch_loss / supervised_sampler.n_batches)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_logits = np.array(all_logits)
        all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

        n_classes = len(np.unique(all_targets))

        acc_total = accuracy_score(all_targets, all_preds)
        precision_total = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall_total = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        mcc_total = matthews_corrcoef(all_targets, all_preds)
        try:
            auc_total = roc_auc_score(all_targets, all_probs, multi_class="ovr")
        except ValueError:
            auc_total = np.nan

        acc_per_class = []
        auc_per_class = []
        mcc_per_class = []
        precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)

        for c in range(n_classes):
            mask = (all_targets == c)
            if mask.sum() > 0:
                acc_c = accuracy_score(all_targets[mask], all_preds[mask])
            else:
                acc_c = np.nan
            acc_per_class.append(acc_c)

            try:
                auc_c = roc_auc_score((all_targets == c).astype(int), all_probs[:, c])
            except ValueError:
                auc_c = np.nan
            auc_per_class.append(auc_c)

            mcc_c = matthews_corrcoef((all_targets == c).astype(int),
                                      (all_preds == c).astype(int))
            mcc_per_class.append(mcc_c)

        metrics_df = pd.DataFrame({
            "ACC": acc_per_class + [acc_total],
            "AUC": auc_per_class + [auc_total],
            "MCC": mcc_per_class + [mcc_total],
            "Precision": list(precision_per_class) + [precision_total],
            "Recall": list(recall_per_class) + [recall_total],
        }, index=[f"class_{i}" for i in range(n_classes)] + ["total"])

        all_metrics.append(metrics_df)

        train_accuracies.append(acc_total)

        all_epoch_results.append({
            "pred_logits": all_probs,   # numpy array, shape=(N, n_classes)
            "preds": all_preds,          # numpy array, shape=(N,)
            "targets": all_targets       # numpy array, shape=(N,)
        })

    return train_losses, train_accuracies, all_metrics, all_epoch_results