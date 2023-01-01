# GLAC(AAAI2023)

This is the MindSpore implementation of GLAC in the following paper.

AAAI 2023: A Generalized Unbiased Risk Estimator for Learning with Augmented Classes

# [GLAC Description](#contents)

1) Propose a generalized URE that can be equipped with arbitrary loss functions while maintaining the theoretical guarantees, given unlabeled data for learning with augmented classes(LAC). 

2) Propose a novel risk-penalty regularization term to alleviate the issue of negative empirical risk commonly encountered by previous studies.

# [Dataset](#contents)

Our experiments are conducted on six regular-scale datasets to test the performance of our GLAC, which are Har, Msplice, Normal, Optdigits, Texture and Usps.

# [Environment Requirements](#contents)

Framework

- [MindSpore](https://gitee.com/mindspore/mindspore)

For more information, please check the resources below：

- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
python main.py
```
