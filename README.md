# One-shot logo detection with keras

Logo detection in images has many applications, such as in advertisement and social media marketing. Existing methods usually require 
huge amount of training data for every logo class, since they are based on standard machine learning paradigms. They learn to recognize logos
from examples of each logo class. Obviously, if you didn't train a model with your logo, you cannot search for it.

For practical application, it is more useful to search for logos in images only providing a query logo (i.e. an example of the logo),
a machine learning paradigm known as "one-shot learning". This is what this repo is about.

This project is an implementation of the paper "A Deep One-Shot Network for Query-based Logo Retrieval" (Bhunia et al. 2019, https://arxiv.org/pdf/1811.01395). The algorithm searches for logos within a given target image, and predicts its presence and location (with a segmentation mask).

The neural network model is composed by a "conditional" branch and a "segmentation" branch. 
The conditional branch gives a latent representation of the query logo. This representation is then combined with feature maps of the segmentation branch at multiple scales, making the model scale-invariant.

![picture alt](https://github.com/giovanniguidi/logo-detection/blob/master/figures/paper.png "")
<center>Illustration of query-based logo detection problem (from Bhunia et al. 2019)</center>

## Depencencies

Install the libraries using:
```
pip install -r requirements.txt 
```


## Weights

The graph and trained weights can be found at:

https://drive.google.com/drive/folders/1ojz6i0dsEEOzJ3qaAeutqD64x5dsxVvF?usp=sharing


## To do
- [x] 


## References


\[1\] [A Deep One-Shot Network for Query-based Logo Retrieval](https://arxiv.org/pdf/1811.01395)


\[2\] [One-Shot Learning introduction (from Medium)](https://connorshorten300.medium.com/one-shot-learning-70bd78da4120)