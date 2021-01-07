# Do we need to freeze embeddings when fine-tuning our LM?

___
An experiment to check the impact of freezing embedding matrix on the validation metric depending on the number of 
non-overlapping tokens between train and validation set.

The experiment was run in the [colab notebook](https://medium.com/r?url=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F167cj0ahu5jtzpCpPvyD0oMdPMy2BzjN6%3Fusp%3Dsharing), 
you can check the tensorboard logs there. There is more detailed explanation in the [article](). 
The main results can be viewed on the following plot: 

![main result plot](https://cdn-images-1.medium.com/max/800/1*oJd76q9m8nMZ1uJzdAR64w.png)