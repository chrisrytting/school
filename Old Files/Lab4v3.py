import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

#Problem 1

campaign_store=pd.HDFStore('campaignstore.h5')
n_chunk = 50000
col_types = {'cmte_id':np.str, 'cand_id':np.str, 'cand_nm':np.str, 'contbr_nm':np.str, 'contbr_city':np.str, 'contbr_st':np.str, 'contbr_zip':np.str, 'contbr_employer':np.str, 'contbr_occupation':np.str, 'contb_receipt_amt':np.float, 'contb_receipt_dt':np.str, 'receipt_desc':np.str,'memo_cd':np.str, 'memo_text': np.str, 'form_tp':np.str, 'file_num':np.float, 'tran_id':np.str, 'election_tp':np.str}
reader = pd.read_csv('campaign.csv', sep=' ', dtype=col_types, min_itemsize=100, parse_dates=['contb_receipt_dt'],index_col=False,chunksize=n_chunk, skipinitialspace=True)
first = True # a flag for the very first chunk
data_cols=['cand_nm', 'contbr_st', 'contbr_occupation', 'contb_receipt_amt', 'contb_receipt_dt'] # queries involving cols B and C will be allowed
for chunk in reader:

	if first:
		campaign_store.append('campaign', chunk, data_columns=data_cols, index=False)
		first = False
		print campaign_store
	else:
		campaign_store.append('campaign', chunk, index=False)

print campaign_store











