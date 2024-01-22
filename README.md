# Effects of Structured Pruning on Handling Uncertainty Estimates
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

<div align = center>
<img width="800" alt="image" src="https://github.com/ksheersagaragrawal/LotteryTicketPruning/assets/76050795/9912143d-aec4-4a37-a4ac-f11ac20586c7">
</div>

--------------------------------------------------------------------------------

# Neural Network Pruning Techniques

## Overview
This repository contains the implementation and experimentation of various neural network pruning techniques, with a focus on structured pruning. The aim is to reduce the complexity of the model while maintaining or improving its general performance and preventing overfitting.

## Pruning Techniques Implemented
This section outlines the various pruning techniques applied in our study referenced from the paper [Lottery Ticket Hypothesis](https://arxiv.org/abs/2003.03033). Iterative Pruning outperforms the rest based on its effectiveness in reducing model complexity while maintaining performance.

- One-Shot Pruning
- Re-Initialised One-Shot Pruning
- Randomly Re-Initialised One-Shot Pruning
- Re-Initialised Iterative Pruning

## Models
- `cifar10.ipynb`: Contains the implementation for the CIFAR-10 (multinary) dataset for **CNN**.
- `make_moon.ipynb`: Demonstrates pruning on the Make Moons (binary) dataset for **FCC**.
- `util.py`: Utility functions used across the models.

## Results and Analysis: 
- **Accuracy** : Estimating model's ability to correctly predict on sample data through test accuracy metrics referenced from the [Lottery Ticket Hypothesis Paper](https://arxiv.org/abs/1803.03635)
  - _Early Stopping Criteria_: Minimum number of epochs required to train the model after Pm percent of pruning. 
  - _Test Accuracy_: Accuracy on the test dataset after Pm percent of pruning.

- **Expected Calibration Error**:  Providing visual insight into the model's calibration before and after pruning referenced from the [Temerature Scaling Paper](https://proceedings.mlr.press/v139/daxberger21a/daxberger21a.pdf)
  - _Reliability Diagram_: Distribution of confidence intervals versus the actual proportion of correct predictions.
  - _ECE_: Expected Caliberation Error after Pm percent of pruning on the test data.
  - _UCE_: Uncertainity Caliberation Error after Pm percent of pruning on the test data.

- **Out-Of-Distribution (OOD) Detection**: Estimating model's ability to correctly predict outcomes on unseen data.
  - _Reliability Diagram_: Examining how the model's predicted confidence levels compare with its actual performance on OOD scenarios.
  - _ECE_: Determining the Expected Calibration Error to quantify the model's predictive confidence accuracy when faced with unfamiliar data.

## Pruning Best Practices
The insights derived from the paper [What is the state of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) and our experiments have been used to formulate a set of best practices for pruning neural networks. 

## References
Refer our paper [Affects of Pruning Neural Network](https://github.com/ksheersagaragrawal/LotteryTicketPruning/blob/main/Affects_of_Pruning_Neural_Network.pdf) included in this repository for an in-depth analysis of the pruning techniques and their impacts on model performance.

## Contributing
For any queries, please open an issue in the repository or contact the maintainers directly. Contributions to this project are welcome. Please send pull requests or open an issue to discuss potential changes or additions.

## Acknowledgements
This project builds upon significant previous work in the field of neural network pruning and would not be possible without the foundational research provided by the following papers and authors:

- Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. Available at [arXiv:1803.03635](https://arxiv.org/abs/1803.03635).
- Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of Neural Network Pruning? Available at [arXiv:2003.03033](https://arxiv.org/abs/2003.03033).
- Bansal, V., Khoiwal, R., Shastri, H., Khandor, H., & Batra, N. (2022). "I do not know": Quantifying Uncertainty in Neural Network Based Approaches for Non-Intrusive Load Monitoring. Available at [Nipun Batra’s publications](https://nipunbatra.github.io/papers/2022/buildsys22-nilm.pdf).
- Daxberger, E., Nalisnick, E., Allingham, J. U., Antoran, J., & Hernández-Lobato, J. M. (2021). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. Available at [ICML Proceedings](https://proceedings.mlr.press/v139/daxberger21a/daxberger21a.pdf).
- Laves, M.-H., Ihler, S., Kortmann, K.-P., & Ortmaier, T. (2021). Well-calibrated Model Uncertainty with Temperature Scaling. Available at [MLR Proceedings](https://arxiv.org/pdf/1909.13550.pdf).

The contributions of Ksheer Agrawal, Lipika Rajpal, and Kanshik Singhal have been invaluable in the development and success of this project.

