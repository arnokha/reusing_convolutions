Code for paper "Reusing Convolutional Activations from Frame to Frame to Speed up Learning and Inference"

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
