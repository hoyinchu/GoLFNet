import torch
from torch.utils.data import Dataset

class ESMDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.covars = ["alt_am_score","is_amplified","amp_del_ratio"]
        self.dataframe[self.covars] = self.dataframe[self.covars].astype(float)
        self.labels = ["GoF","LoF","Neutral"]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sequence = row['sequence']
        covar_arr = row[self.covars].values.astype(float)
        covariates = torch.tensor(covar_arr,dtype=torch.float32)#.view(-1,1)
        label_arr = row[self.labels].values.astype(float)
        labels = torch.tensor(label_arr,dtype=torch.float32)
        return sequence, covariates, labels