# This script gathers a few random statements from the speeches and then submits to openai to get topic distribution for each statment
#%%
import numpy as np
import pandas as pd
import openai
import tiktoken
import json
#%%
# commented out for safety
#with open("../../secrets.json",'r') as f:
#    openai.api_key=json.load(f)["OPENAI_API_KEY"]

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
        {"role":"system","content":"You are a helpful assistant"},
        {"role":"user","content":"Consider the following statement"},
        {"role":"user","content":f"{msg}"},
        {"role":"user","content":f"On a single line, write a comma-separated list of 5 keywords that describe this statement. "}
    ]
    return prompt

prompt=mk_prompt('A second way in which banks have been deemed to be "special" is in the provision of basic banking services such as credit extension, deposit-taking, and payments processing.   There is little question that these functions are critically important throughout society.  Consumers turn to banks for safe investments such as time and savings deposits, for')

print(f"this message has {count_tokens(prompt)} tokens")




#%%
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=prompt)

print(completion.choices[0].message.content)


# %%
x=pd.read_csv("./data/speeches/fed_speeches_paragraphs.csv")

resplist=[]
sample=np.random.choice(x.index,20)

#%%
for i in sample:
    prompt=mk_prompt(x.loc[i,"text"])
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=prompt)
    resp=json.loads(json.dumps(resp))
    resp["prompt"]=prompt
    resp["observation_index"]=f"{i}"
    resplist.append(resp)

#%%
with open("test_out.jsonl","w") as f:
    for i in resplist:
        f.write(json.dumps(i)+"\n")
# %%
