Code for paper "Reusing Convolutional Activations from Frame to Frame to Speed up Learning and Inference"

#### Acknowledgements:
[Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/) for main training loop  
[Pytorch contributors](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py) REINFORCE functions select_action() and updade_policy()/finish_episode()

### Results

#### Inference: time to run 3000 steps, grayscale, downsampled by a factor of 4 in each dimension

Number of filters in convolutional layers one and two: 20/40
![ID4_20_40](figures/inference_d4_20_40_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 40/80
![ID4_40_80](figures/inference_d4_40_80_heatmap_no_cbar.png)

Number of filters in convolutional layers one and two: 80/160
![ID4_80_160](figures/inference_d4_80_160_heatmap_no_cbar.png)


#### Inference: time to run 3000 steps, grayscale, downsampled by a factor of 2 in each dimension

Number of filters in convolutional layers one and two: 20/40

Number of filters in convolutional layers one and two: 40/80

Number of filters in convolutional layers one and two: 80/160


#### Training: time to run 3000 steps, grayscale, downsampled by a factor of 4 in each dimension

Number of filters in convolutional layers one and two: 20/40

Number of filters in convolutional layers one and two: 40/80

Number of filters in convolutional layers one and two: 80/160


#### Training: time to run 3000 steps, grayscale, downsampled by a factor of 2 in each dimension

Number of filters in convolutional layers one and two: 20/40

Number of filters in convolutional layers one and two: 40/80

Number of filters in convolutional layers one and two: 80/160
