# STULIG
data source for STULIG

Towards Interpreting, Discriminating and Synthesizing Motion Traces via Probabilistic Generative Models

# Environment
* python 2.7
* Tensorflow 1.7 or ++ （updated now 2019.06)
# Dataset
Here we list a gowalla dataset for training. 
* Gowalla: http://snap.stanford.edu/data/loc-gowalla.html
* (remark) Please do not use these datasets for commercial purpose. For academic uses, please cite the paper. Thanks for their help.
# Usage
* word2vec: Use the pip tool 'pip install word2vec'，The data tha we have removed the POIs which the frequency is less than a threshold. For pure semi-supervised learning (never know which are test data) which we have proposed in our paper, embedding all of the POIs including all of the labelled data and unlabelled data (just like scopus). (Important: remove the user ID from gowalla_scopus_1104.dat) command as: 'word2vec -train gowalla_scopus_real.dat output gowalla_em_250.dat -size 250 -window 5 -min-count 0 -cbow 0'
* Training process: We choose the 201 users' sub-trajectories, split these to training data(about 50%) and test data (about 50%). The new code with tensorflow>=1.7, you can run it easily. and also some records will stored by the code (including model, train data and sample results)
* The format of total.dat : userid/locationid/time/longitude/latitude
* STUL used the CNN to replace RNN in encoder and decoder. STULIG added the extra time and space information to improve the performance of the experient.

# Keynotes

The main differences between this article and the previous ones are as follows: firstly, information on time and space is added to the classification process so that the whole classification process is more accurate; Secondly, we use CNN to replace the RNN used in most methods before, which not only achieves faster training time, but also achieves good training effect.

# Reference
Hope such an implementation could help you on your projects. Any comments and feedback are appreciated.
