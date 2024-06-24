# EML_Pruning_ADMM
Final Project for Embedded Machine Learning of Heidelberg University. Group5 task is replicate the ADMM pruning and implement it specific LeNet-5 and AlexNet.

Management Document:
https://www.notion.so/fffrozen/Embedded-Machine-Learning-Final-Project-16034c82d99a41a0b3cefdaef3a72d8c?pvs=4


Dicionary Structure:
- `src/` : Contains the source code for the project
    - `ADMM_Toolbox/` : Contains the ADMM toolbox
    - `SimplePruning_Toolbox/`(Optional) : Contains the simple pruning toolbox such as 1. Enegy Efficiency-Aware Pruning, 2. Structured-Preserved Pruning. These methods are not major task of the project, but it is good comparison with ADMM.
    - `LeNet5/` : Contains the LeNet5 implementation
    - `AlexNet/` : Contains the AlexNet implementation
- `data/` : Contains the data for the project
- `results/` : Contains the results of the project
- `docs/` : Contains the documentation of the project
- `references/` : Contains the references of the project
- `docker/` : Contains the docker files for the project


TODOs:
Implement AlexNet for ImageNet
Implement LeNet5 for MNIST
Implement ADMM Pruning for LeNet5
Implement ADMM Pruning for AlexNet

