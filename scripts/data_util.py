import pandas as pd
import numpy as np

def mut_to_seq(seq,pos,ref,alt):
    try:
        if seq[pos-1]==ref:
            return seq[:pos-1] + alt + seq[pos:]
        else:
            print("Reference amino acid mismatch. This could be an indexing issue (amino acid sequence index starts at 1) or the data is using a non-canonical transcript. Returning nan")
            return np.NaN
    except:
        #print(e)
        print("Error encountered likely due to specified aa_pos exceeding reference position. Returning NaN")
        return np.NaN