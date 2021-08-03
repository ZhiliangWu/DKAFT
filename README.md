# Description


The Python implementation for the publication[ “Uncertainty-Aware Time-to-Event Prediction using Deep Kernel Accelerated Failure Time Models”](https://arxiv.org/abs/2107.12250) on [MLHC 2021](https://www.mlforhc.org/) (Machine Learning for Healthcare 2021) using PyTorch and GPyTorch packages. 

# Project structure

```bash
.
├── data_utils.py
├── gp_layer.py
├── LICENSE
├── logging_conf.py
├── model_utils.py
├── plot_utils.py
├── pml_los.py
├── pml_pfs.py
├── pytorchtools.py
├── README.md
├── requirements.txt
├── run_exp_gp_los_metric.py
├── run_exp_gp_los.py
├── run_exp_gp_pfs_exact_metric.py
├── run_exp_gp_pfs_exact.py
├── run_exp_gp_pfs_svgp_metric.py
├── run_exp_gp_pfs_svgp.py
├── run_exp_los_dropout.py
├── run_exp_los_metric.py
├── run_exp_los.py
├── run_exp_pfs_dropout.py
├── run_exp_pfs_metric.py
└── run_exp_pfs.py

```

# Usage 

* The access to datasets for the prediction of Progression Free Survival (PFS) and Length-of-Stay (LoS) has to be applied before running the code. 
    * THe prediction of PFS involves the [PRAEGNANT Dataset](http://www.praegnant.org/). 
    * The prediction of LoS involves the [MIMIC dataset](https://mimic.mit.edu/iii/gettingstarted/) with the data pre-processing from [Purushotham et al.](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII).
* All `.py` files should be able to run with `python xxx.py` after installing the packages specified in `requirements.txt`.
* The `.py` scripts prefixed with `run_exp_` can be used to build models proposed in [the paper](https://arxiv.org/abs/2107.12250).
    * Scripts with `…_pfs_…` are for the task of PFS prediction.
    * Scripts with `…_los_…` are for the task of LoS prediction. 

# Note

The code is published to ensure the reproducibility in the machine learning community. If you find the code helpful, please consider citing 

```bib
@article{wu2021uncertainty,
  title={Uncertainty-Aware Time-to-Event Prediction using Deep Kernel Accelerated Failure Time Models},
  author={Wu, Zhiliang and Yang, Yinchong and Fasching, Peter A and Tresp, Volker},
  journal={arXiv preprint arXiv:2107.12250},
  year={2021}
}
```


# License 

The code has a MIT license, as found in the [LICENSE](./LICENSE) file.