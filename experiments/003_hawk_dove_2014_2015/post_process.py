# this loads in the processed data and reconstructs the original dataset with the topic labels attached

#%%
import numpy as np
import pandas as pd
import openai
import tiktoken
import json
import os,sys
import re
# config
sample=False

#%%
if sample:
    json_file="hawk_dove_responses_sample.jsonl"
else:
    json_file="hawk_dove_responses.jsonl"

if os.path.exists(json_file):
    resplist=[]
    with open(json_file,"r") as f:
        lines=f.readlines()
        for i in lines:
            resplist.append(json.loads(i))


# %%


hawkframe=pd.DataFrame([{"index":i["observation_index"],"topics":i["choices"][0]["message"]["content"]} for i in resplist])
hawkframe.loc[:,"index"]=hawkframe.index.astype("int")

hawkframe.loc[:,"topics"]=hawkframe.loc[:,"topics"].apply(lambda a: re.sub("^[\n\s]*","",a))


hawkframe.loc[:,"rating"]=np.nan

rate_pattern=re.compile("^.*?(NA|\d)")
rate_pattern2=re.compile("^Answer:.*?\n\n(NA|\d)")
def get_pattern(a,pat):  
    m=re.match(pat,a)
    if m:
        return m.groups()[0]
    else:
        return None


msk=hawkframe.loc[:,"rating"].isna()
hawkframe.loc[msk,"rating"]=hawkframe.loc[msk,"topics"].apply(lambda a:get_pattern(a,rate_pattern))

msk=hawkframe.loc[:,"rating"].isna()
hawkframe.loc[msk,"rating"]=hawkframe.loc[msk,"topics"].apply(lambda a:get_pattern(a,rate_pattern2))

msk=hawkframe.loc[:,"rating"].isna()
print("still fix these:")
hawkframe.loc[msk,:]
#%%
# split out explainations
hawkframe.loc[:,"explain"]=hawkframe.loc[:,"topics"]

hawkframe.loc[:,"explain"]=hawkframe.loc[:,"explain"].apply(lambda a: re.sub("^[(NA|\d|Rating:|Answer:|Explanation:)\n\s]*","",a))

hawkframe.loc[:,"explain"]=hawkframe.loc[:,"explain"].apply(lambda a: re.sub("^.+?\n\b","",a))


#%%
x=pd.read_csv("speeches.csv",index_col=0) # specific version used for this data
#%%
x=pd.merge(hawkframe,x,left_on="index",right_index=True)
#%%
x.drop(columns=["index","idx"],inplace=True)

#%%
outname="speeches_with_hawk_dove.csv"
x.to_csv(outname)


