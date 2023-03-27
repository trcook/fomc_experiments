# This script gathers statements from speeches between 2014 and 2015 then submits to openai to get topic distribution for each statment. This is useful for working out what should be used for the hawk-dove experiment
#%%
import numpy as np
import pandas as pd
import openai
import tiktoken
import json

################

# This script gathers all statements from the speeches and then submits to openai to get topic distribution for each statment
#%%
import numpy as np
import pandas as pd
import openai
import tiktoken
import json
import os,sys
#%%

#%% 
# config
sample=False
start_year=2014
end_year=2015

if os.path.exists("./speeches.csv"):
    x=pd.read_csv("./speeches.csv",index_col=0)
else:
    x=pd.read_csv("./data/speeches/fed_speeches_paragraphs.csv")
    x.loc[:,"merge_index"]=x.index
    msk=(x.year>=start_year) & (x.year<=end_year)
    x=x.loc[msk,:]
    x.reset_index(inplace=True,drop=True)
    x.to_csv("speeches.csv")

#%%
# commented out for safety
with open("../../secrets.json",'r') as f:
   openai.api_key=json.load(f)["OPENAI_API_KEY"]

#%%
models = openai.Model.list()


#%%
# count tokens function

def count_tokens(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

# %%
def mk_prompt(msg):
    prompt=[
        {"role":"system","content":"You are an economist"},
        {"role":"user","content":"Consider the following statement"},
        {"role":"user","content":f"{msg}"},
        {"role":"user","content":f"On a single line, rate the statement from 1 to 10 where 1 is very dovish and 10 is very hawkish; write 'NA' if the statement is neither hawkish or dovish. On a new line, explain your reasoning in one sentence."}
    ]
    return prompt



restart=True
while restart==True:
    try:
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

            remainder_index=list(set(x.index)-set([int(i["observation_index"]) for i in resplist]))
        else:
            remainder_index=x.index

        if sample:
            remainder_index=np.random.choice(remainder_index,20)

        resplist=[]

        for i in remainder_index:
            prompt=mk_prompt(x.loc[i,"text"])
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=prompt,max_tokens=100)
            resp=json.loads(json.dumps(resp))
            resp["prompt"]=prompt
            resp["observation_index"]=f"{i}"
            resplist.append(resp)
            with open(json_file,"a") as f:
                f.write(json.dumps(resp)+"\n")
        restart=False
        break
    except Exception as e:
        restart=True
        print(e)



#%%

