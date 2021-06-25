## Changes due to stance labels

## TODO
1. Run code with stance data - Use flag
 - Pivots
- classifier 3 labels
- make sure bert does not use labels
2. Fix pivots by Refael
3. Full run something




Main Question - 
what is the input?
Supposedly, 2 texts, tweet and domain.


Necessary because label relates between tweet and domain. 
I.e. What is the stance of the tweet in regards to the topic (domain)? 
On the other hand, for a given domain, the topic is the same for all tweets. 

Benefits of not inserting the topic to the model:
- Easier to implement

Benefits of inserting the topic to the model:
-   

Negatives of not inserting the topic to the model:
- For different domains, generalization is questionable as the labels are derived from the topic as well.

Negatives of inserting the topic to the model:
- Same input to all tweets in domain. What is the added value?

0. General
- mode flag
- Paths to data

1. pivot selection
   - recreate pivots with stance labels
   - 

2. perl_pretrain - finetune bert

  
3. supervised_task_learning  Classification



For Roi -
does it make sense to concat tweet and topic? any better way?