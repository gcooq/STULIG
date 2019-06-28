# STULIG
The implementation for STULIG

Towards Interpreting, Discriminating and Synthesizing Motion Traces via Deep Probabilistic Generative Models

# Environment
* python 2.7
* Tensorflow 1.7 or ++ （updated now 2019.06)
# Dataset
Here we list a gowalla dataset for training. 
* Gowalla: http://snap.stanford.edu/data/loc-gowalla.html
* (remark) Please do not use these datasets for commercial purpose. For academic uses, please cite the paper. Thanks for their help.
# Usage
1. Training process: We choose the 201 users' sub-trajectories, split these to training data(about 50%) and test data (about 50%). The new code with tensorflow>=1.7, you can run it easily. and also some records will stored by the code (including model, train data and sample results)
2. The format of total.dat : userid/locationid/time/longitude/latitude
* *python STUL.py*
* *python STULIG.py*

# Reference
Hope such an implementation could help you on your projects. Any comments and feedback are appreciated.
