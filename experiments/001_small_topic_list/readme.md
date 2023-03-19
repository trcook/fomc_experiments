# This experiment

We send the following prompt to chatGPT:

>system: You are a helpful assistant
>user: Consider the following statement
>{{MSG}}
>On a single line, write a comma-separated list of 5 keywords that describe this statement.

This was run for an old version of the data and, consequentially, only generates results for 1996-1998. Running the code again should basically fix it since the underlying data was fixed. 

The file experiments/001_small_topic_list/all_speeches_with_topics.csv has topics labeled for 1996-1998. It is setup for easy reading as a csv