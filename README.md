# Do we need to freeze embeddings when fine-tuning our LM?

An experiment to check the impact of freezing embedding matrix on the validation metric depending on the number of 
non-overlapping tokens between train and validation set.

The experiment was run in the [colab notebook](https://colab.research.google.com/drive/167cj0ahu5jtzpCpPvyD0oMdPMy2BzjN6?usp=sharing),
you can check the tensorboard logs there. There is more detailed explanation in the [article](https://korenv20.medium.com/do-we-need-to-freeze-embeddings-when-fine-tuning-our-lm-c8bccf4ffeba). 
The main results can be viewed on the following plot: 

![main result plot](https://cdn-images-1.medium.com/max/800/1*oJd76q9m8nMZ1uJzdAR64w.png)
