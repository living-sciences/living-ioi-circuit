This repository contains the code for all experiments in the paper "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small" (Wang et al, 2022).

<img src="https://i.imgur.com/iptFVBc.png">

This is intended as a one-time code drop. <b>The authors recommend those interested in mechanistic interpretability use the <a href="https://github.com/TransformerLensOrg/TransformerLens">Transformer Lens</a> library</b>. 

Specifically, <a href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb">this TransformerLens demo</a> goes through a number of experiments from the Interpretability in the Wild paper, and also introduces other features of that library, which are helpful for building off of our research.

Contact arthurconmy@gmail.com or comment on <a href="https://github.com/redwoodresearch/Easy-Transformer/pull/8">this</a> PR (sadly issues don't work for forks) for proposed changes.

# Quick Start

See and run the experiments <a href="https://colab.research.google.com/drive/1n4Wgulv5ev5rgRUL7ypOw0odga9LEWHA?usp=sharing">on Google Colab</a>.

# Setup

## Option 1) install with pip

```
pip install git+https://github.com/redwoodresearch/Easy-Transformer.git
```

## Option 2) clone repository (for development, and finer tuning)

```bash
git clone https://github.com/redwoodresearch/Easy-Transformer/
pip install -r requirements.txt
```

# In this repo

In this repo, you can find the following notebooks (some are in `easy_transformer/`):

* `experiments.py`: a notebook of several of the most interesting experiments of the IOI project.
* `completeness.py`: a notebook that generate the completeness plots in the paper, and implements the completeness functions.
* `minimality.py`: as above for minimality.
* `advex.py`: a notebook that generates adversarial examples as in the paper.
`
# Easy Transformer

<i>(later renamed "TransformerLens")</i>

## An implementation of transformers tailored for mechanistic interpretability.

It supports the importation of open sources models, a convenient handling of hooks to get access to intermediate activations and features to perform simple emperiments such as ablations and patching.

A demo notebook can be found [here](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb), with links to other tutorials and demos too.
