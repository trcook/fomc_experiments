# this loads in the processed data and reconstructs the original dataset with the topic labels attached

#%%
import numpy as np
import pandas as pd
import openai
import tiktoken
import json
import os,sys


#%%
if os.path.exists("out_all.jsonl"):
    resplist=[]
    with open("out_all.jsonl","r") as f:
        lines=f.readlines()
        for i in lines:
            resplist.append(json.loads(i))


# %%
