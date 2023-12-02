import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, EsmModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve,auc
import timeit
from tqdm import tqdm

def run_one_epoch(
    train_flag,
    dataloader,
    esm_combined_model,
    optimizer,
    device="cuda"
):
    """
    Run one epoch of the model and report relevant training metrics
    """
    torch.set_grad_enabled(train_flag)
    esm_combined_model.train() if train_flag else esm_combined_model.eval()
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for idx, (sequences, covariates, labels) in enumerate(tqdm(dataloader)):
        covariates, labels = covariates.to(device), labels.to(device)
        output = esm_combined_model(sequences,covariates) # forward pass
        output = output.to(device)
        # output = output.squeeze() # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits(output,labels) # numerically stable

        if train_flag:
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        predictions = (output >0.0).float()
        accuracy = torch.mean((predictions == (labels > 0.5)).float())
        accuracies.append(accuracy.detach().cpu().numpy())

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu().numpy(), predictions.cpu().numpy(), average=None)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # accuracy = torch.mean( ( (output > 0.0) == (labels > 0.5) ).float() ) # output is in logit space so threshold is 0.
        
        #accuracies.append(accuracy.detach().cpu().numpy())
    
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions, axis=0)
    avg_recall = np.mean(recalls, axis=0)
    avg_f1_score = np.mean(f1_scores, axis=0)

    return (avg_loss,avg_accuracy,avg_precision,avg_recall,avg_f1_score)    
    #return( np.mean(losses), np.mean(accuracies) )

def train_model(
    esm_combined_model,
    train_esm_dataloader,
    val_esm_dataloader,
    epochs=100,
    patience=10,
    checkpoint_name=None,
):
    """
    Train a esm_combined_model model and record accuracy metrics.
    """
    # Move the model to the GPU here to make it runs there, and set "device" as above
    # TODO CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_combined_model.to(device)
    print(f"Device: {device}")
    
    # 1. Make new BedPeakDataset and DataLoader objects for both training and validation data.
    # train_esm_dataset = ESMDataset(train_data)
    # val_esm_dataset = ESMDataset(validation_data)    
    # train_esm_dataloader = torch.utils.data.DataLoader(train_esm_dataset, batch_size=batch_size, num_workers = 0)
    # val_esm_dataloader = torch.utils.data.DataLoader(val_esm_dataset, batch_size=batch_size, num_workers = 0)
    
    # 2. Instantiates an optimizer for the model.
    optimizer = torch.optim.Adam(esm_combined_model.parameters(), amsgrad=True)
    
    # 3. Run the training loop with early stopping.
    metric_df_rows = []
    #train_accs = []
    #val_accs = []
    patience_counter = patience
    best_val_loss = np.inf
    if checkpoint_name:
        check_point_filename = checkpoint_name
    else:
        check_point_filename = 'model_checkpoints/checkpoint.pt' # to save the best model fit to date
    print("Epoch starting")
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        train_loss, train_acc, train_prec, train_recall, train_f1 = run_one_epoch(True, train_esm_dataloader, esm_combined_model, optimizer, device)
        val_loss, val_acc, val_prec, val_recall, val_f1 = run_one_epoch(False, val_esm_dataloader, esm_combined_model, optimizer, device)
        
        metric_df_rows.append({
            "epoch": epoch,
            "avg_loss":train_loss,
            "avg_acc":train_acc,
            "avg_prec_GoF":train_prec[0],
            "avg_recall_GoF": train_recall[0],
            "avg_f1_GoF": train_f1[0],
            "avg_prec_LoF":train_prec[1],
            "avg_recall_LoF": train_recall[1],
            "avg_f1_LoF": train_f1[1],
            "avg_prec_Neutral":train_prec[2],
            "avg_recall_Neutral": train_recall[2],
            "avg_f1_Neutral": train_f1[2],
            "setting": "training"
        })
        metric_df_rows.append({
            "epoch": epoch,
            "avg_loss":val_loss,
            "avg_acc":val_acc,
            "avg_prec_GoF":val_prec[0],
            "avg_recall_GoF": val_recall[0],
            "avg_f1_GoF": val_f1[0],
            "avg_prec_LoF":val_prec[1],
            "avg_recall_LoF": val_recall[1],
            "avg_f1_LoF": val_f1[1],
            "avg_prec_Neutral":val_prec[2],
            "avg_recall_Neutral": val_recall[2],
            "avg_f1_Neutral": val_f1[2],
            "setting": "validation"
        })
        if val_loss < best_val_loss:
            #torch.save(esm_combined_model.state_dict(), check_point_filename)
            torch.save({
                'model_state_dict': esm_combined_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, check_point_filename)
            
            best_val_loss = val_loss
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                esm_combined_model.load_state_dict(torch.load(check_point_filename)["model_state_dict"]) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        print(f"Epoch {epoch} took {elapsed}s")
        print(f"Latest stats:")
        print(metric_df_rows[-2])
        print(metric_df_rows[-1])
        print(f"Patience left: {patience_counter}")
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))
    
    metric_df = pd.DataFrame(metric_df_rows)
    metric_df.to_csv(f"{check_point_filename}.log.tsv",sep="\t",index=False)
    
    # 4. Return the fitted model (not strictly necessary since this happens "in place") and a dataframe that kept track of all the training/validation acc/losses
    return (esm_combined_model, metric_df)

## Wrapper function for training sklearn models
def train_sklearn_model(
    model,
    train_data,
    test_data,
    covar_set
):
    """
    Train logistic regression model gigven the training and test data and output the trained model
    as well as its predictions
    """
    # Process the training and testing data
    train_X = train_data[covar_set]
    train_y = train_data["pred_col"].to_numpy()
    
    test_X = test_data[covar_set]
    test_y = test_data["pred_col"].to_numpy()
    
    # Fit the model
    model.fit(train_X, train_y)
    
    # Make prediction
    predicted_output = model.predict_proba(test_X)
    pred_output_df = pd.DataFrame(columns=["pred_GoF","pred_LoF","pred_Neutral"],data=predicted_output)
    pred_output_df["max_pred"] = pred_output_df[["pred_GoF","pred_LoF","pred_Neutral"]].idxmax(axis=1)
    pred_output_df["true_label"] = test_y
    
    # Return a new test df with the predicted score
    test_data_new = pd.concat([test_data,pred_output_df],axis=1)
    return model,test_data_new

## Evaluate model performance
def eval_model_performance(df,model_type,eval_setting):
    """
    Given a dataframe with prediction outputs, calculate its performances in GoF/LoF/Neutral in AUROC and AUPRC
    """
    eval_rows = []
    df["max_pred"] = df["max_pred"].str.replace("pred_","")
    for label in ["GoF","LoF","Neutral"]:
        y =  (df["aa_change_category"] == label).astype(int).to_numpy()
        scores = df[f"pred_{label}"].to_numpy()
        hard_pred = (df["max_pred"]==label).astype(int).to_numpy()
        precision, recall, auprc_thresholds = precision_recall_curve(y,scores)
        auprc = auc(recall, precision)
        hard_precision, hard_recall, hard_f1, _ = precision_recall_fscore_support(y,hard_pred, average="binary")

        exception = None
        try:
            auroc = roc_auc_score(y,scores)
        except ValueError:
            auroc = np.NaN
            exception = "ValueError"
        eval_rows.append({
            "label":label,
            "label_pos": y.sum(),
            "label_neg": len(y)-y.sum(),
            "auroc":auroc,
            "auprc":auprc,
            "precision": hard_precision,
            "recall": hard_recall,
            "f1_score": hard_f1,
            "exception": exception,
        })
    return_df = pd.DataFrame(eval_rows)
    return_df["model_type"] = model_type
    return_df["eval_setting"] = eval_setting
    return return_df

## Given model predictions, stratified the performnace metrics by each individual holdout genes
## then lump the rest together
def eval_model_performance_stratified(df,holdout_genes,model_type):
    performance_dfs = []
    for gene in holdout_genes:
        sub_df = df[df["Hugo_Symbol"]==gene]
        performance_df = eval_model_performance(sub_df,model_type,gene)
        performance_dfs.append(performance_df)
    partial_performance_df = eval_model_performance(df[~df["Hugo_Symbol"].isin(holdout_genes)],model_type,"Partial")
    performance_dfs.append(partial_performance_df)
    performance_df = pd.concat(performance_dfs)
    return performance_df

## Evaluate model performance on a by gene basis
def eval_model_performance_per_gene(df,model_type):
    gene_performances = []
    for gene, gene_df in df.groupby("Hugo_Symbol"):
        performance = eval_model_performance(gene_df,model_type)
        performance["gene"] = gene
        gene_performances.append(performance)
    return_df = pd.concat(gene_performances)
    return_df["model_type"] = model_type
    return return_df

## Given a trained model and dataloader make predcitons on samples in the dataloader using the given model
def make_prediction(trained_model,dataloader):
    device = "cuda"
    outputs = []
    trained_model = trained_model.eval().to(device)
    torch.set_grad_enabled(False)
    output_dfs = []

    for idx,batch in enumerate(tqdm(dataloader)): # iterate over batches\
        sequences, covariates, labels = batch
        covariates = covariates.to(device)
        output = trained_model(sequences, covariates)
        output = torch.sigmoid(output)
        output_np = output.detach().cpu().numpy()
        output_df = pd.DataFrame(output_np,columns=["pred_GoF","pred_LoF","pred_Neutral"])
        #output_df[["GoF_label","LoF_label","Neutral_label"]] = labels
        output_df["max_pred"] = output_df[["pred_GoF","pred_LoF","pred_Neutral"]].idxmax(axis=1)
        output_df["sequence"] = sequences
        output_dfs.append(output_df)
    combined_outputs = pd.concat(output_dfs)
    return combined_outputs