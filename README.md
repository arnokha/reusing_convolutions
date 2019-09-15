Code for paper ["Reusing Convolutional Activations from Frame to Frame to Speed up Learning and Inference"](https://arxiv.org/abs/1909.05632)

#### Acknowledgements:
[Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/) for main training loop  
[Pytorch contributors](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py) REINFORCE functions select_action() and updade_policy()/finish_episode()

### Results - Atari
Results are averaged over 10 runs. The policy network used for the experiments consists of 2 convolutional layer followed by a dense layer.

#### Inference: time to run 3000 steps (in seconds), grayscale, downsampled by a factor of 4 in each dimension

Number of filters in convolutional layers one and two: 20/40
![ID4_20_40](figures/inference_d4_20_40_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 40/80
![ID4_40_80](figures/inference_d4_40_80_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 80/160
![ID4_80_160](figures/inference_d4_80_160_heatmap_no_cbar.png)


#### Inference: time to run 3000 steps (in seconds), grayscale, downsampled by a factor of 2 in each dimension

Number of filters in convolutional layers one and two: 20/40
![ID2_20_40](figures/inference_d2_20_40_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 40/80
![ID2_40_80](figures/inference_d2_40_80_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 80/160
![ID2_80_160](figures/inference_d2_80_160_heatmap_no_cbar.png)



#### Training: time to run 3000 steps (in seconds), grayscale, downsampled by a factor of 4 in each dimension

Number of filters in convolutional layers one and two: 20/40
![TD4_20_40](figures/training_d4_20_40_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 40/80
![TD4_40_80](figures/training_d4_40_80_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 80/160
![TD4_80_160](figures/training_d4_80_160_heatmap_no_cbar.png)


#### Training: time to run 3000 steps (in seconds), grayscale, downsampled by a factor of 2 in each dimension

Number of filters in convolutional layers one and two: 20/40
![TD2_20_40](figures/training_d2_20_40_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 40/80
![TD2_40_80](figures/training_d2_40_80_heatmap_no_cbar.png)

### Requirments to Run
Requires:  
numpy  
torch  
torchvision  
gym[atari]  
jupyter notebook is required to view and run the notebooks in the root directory, but is not required for running performance tests.

### Reproducibility
**To rerun tests (with modification)**:

_Editing number of filters in each layer_: Download the repo, and edit the bash scripts located in each subfolder (e.g. "reusing_convolutions/performance_tests/INFERENCE_D2/REUSE_CPU/run_performance_tests_2_layers_1.sh"). You can run those tests individually or run them all at once from the parent directory.

_Editing architecture further and other options_: Download the repo, and edit the python file located in each subfolder (e.g. "reusing_convolutions/performance_tests/INFERENCE_D2/REUSE_CPU/test_atari_2_layer_reuse_cpu.py").

**To rerun tests (without modification)**: 

Download the repo, and run the bash script in each experiment category directory (e.g. "reusing_convolutions/tree/master/performance_tests/INFERENCE_D2/run_tests.sh")  
Alternatively, to run specific tests, you can edit and run scripts in the subfolders.
