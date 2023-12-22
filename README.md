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
- Early Stopping Criteria
- One-Shot Pruning
- Re-Initialised One-Shot Pruning
- Randomly Re-Initialised One-Shot Pruning
- Iterative Pruning

## Models
- `cifar10.ipynb`: Contains the implementation for the CIFAR-10 dataset.
- `make_moon.ipynb`: Demonstrates pruning on the Make Moons dataset.
- `util.py`: Utility functions used across the models.

## Results and Analysis
The experiments conducted explore the effects of pruning on accuracy, expected calibration error, and out-of-distribution (OOD) detection.

## Pruning Best Practices
Based on the results, best practices for pruning neural networks are discussed to guide future work in the field.

## How to Use
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt` (if available).
3. Run the Jupyter Notebooks to train and prune the models.

## References
Refer to the paper "Affects of Pruning Neural Network" included in this repository for an in-depth analysis of the pruning techniques and their impacts on model performance.

## Contributing
Contributions to this project are welcome. Please send pull requests or open an issue to discuss potential changes or additions.

## License
Specify the license under which your project is made available.

## Contact
For any queries, please open an issue in the repository or contact the maintainers directly.

## Acknowledgements
- Authors of the referenced paper and datasets.
- Contributors to the project.

