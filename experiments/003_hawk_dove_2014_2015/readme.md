# This experiment

For 2014 and 2015 speeches, we send the following prompt to chatGPT:

>system: You are an economist
>user: Consider the following statement
>{{MSG}}
>On a single line, write a comma-separated list of 5 keywords that describe this statement.

This runs through "topic_label.py"

Then we run "hawk_dove.py"

This filters to only statments that involve monetary policy and then asks to label hawk and dove.

>system: You are an economist
>user: Consider the following statement
>{{MSG}}
>On a single line, rate the statement from 1 to 10 where 1 is very dovish and 10 is very hawkish. On a new line, explain your reasoning in one sentence. 
