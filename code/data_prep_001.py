# split data into paragraph chunks and reset into large dataframe
#%%
import pandas as pd
import tiktoken

encoding=tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
#%%
dat=pd.read_csv("../data/speeches/fed_speeches_1996_2020.csv")

# %%
# filter down to one year

# split into rows with one row per paragraph
other_cols=set(dat.columns)-{"text"}
#for i in dat[i,:].text.split("n"):
#%%
# fix up the data
def preprocess(i):
    i=i.replace("\r","")
    for j in range(10):
        i=i.replace("  "," ")
        i=i.replace("\n ","\n")
    for j in range(3):
        i=i.replace("\n\n","\n")
    i=i.replace("\t","")
    return i

dat.loc[:,"text"]=dat.text.apply(preprocess)



#%%
def split_and_merge(i):
    other_cols=set(dat.columns)-{"text"}
    out=pd.DataFrame({"text":i.text.split("\n")})
    out.loc[:,"idx"]=1
    # out=out.loc[lambda a:len(encoding.encode(a.text))>10 a.text.str.len()>20,:]
    out.loc[:,"num_tokens"]=out.text.apply(lambda a: len(encoding.encode(a)))
    msk=out.num_tokens>25
    out=out.loc[msk,:]
    out.reset_index(inplace=True,drop=True)

    out.loc[:,"paragraph_id"]=out.index.to_list()
    
    i.loc["idx"]=1
    i=i.to_frame().transpose()
    #i= pd.DataFrame(i.to_dict(),index=[0])
    out=pd.merge(i.loc[:,list(other_cols.union({"idx"}))],out,on="idx")
    return out



# %%
out=dat.apply(split_and_merge,axis=1).tolist()
# %%
out=pd.concat(out)
# %%

out.to_csv("../data/speeches/fed_speeches_paragraphs.csv",index=False)
# %%
# to read and check
x=pd.read_csv("../data/speeches/fed_speeches_paragraphs.csv")
# %%

# #%%
