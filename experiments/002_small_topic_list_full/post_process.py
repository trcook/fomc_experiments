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


topicframe=pd.DataFrame([{"index":i["observation_index"],"topics":i["choices"][0]["message"]["content"]} for i in resplist])
topicframe.loc[:,"index"]=topicframe.index.astype("int")
#%%
x=pd.read_csv("../../data/speeches/fed_speeches_paragraphs.csv") # specific version used for this data
#%%
x=pd.merge(topicframe,x,left_on="index",right_index=True)
#%%
x.drop(columns=["index","idx"],inplace=True)




#%% breakout topics into columns. 
# FWIW -- topics appear to be listed in descending order from most important to least -- though sometimes that's not true
def split_topics(i,n):
    il=i.split(",")
    if len(il)<(n+1):
        return ""
    else: 
        return il[n].strip()

for n in range(5):
    x.loc[:,f"topic{n+1}"]=x.loc[:,"topics"].apply(lambda i: split_topics(i,n))


#%% save back again


x.to_csv("./all_speeches_with_topics.csv")
#%% get some data about 
# just looking at soem quick things about topic distribution


stopics=pd.concat([x.loc[:,f"topic{i}"] for i in range(1,6)])
stopics=stopics.str.lower()
stopics=stopics.str.replace(".","")
stopics=pd.value_counts(stopics)

stopics.to_csv("topic_frequencies.csv")
stopics[:100].to_csv("topic_frequencies_top100.csv")