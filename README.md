# Synthetic-spatial-data-generation
This repository contains the materials and documentation for the implementation of probabilistic conditional diffusion model, DriffTraj framework from Zhu et. al. (2023) with weather information to generate high-quality synthetic trajectory data under different underlying distribution in non-IID and OOD scenario.

### Dependencies
Before running the script, ensure that you have following the prerequisites installed:

Programming Language  : Python 3.11.1

Libraries :
* numpy                 
* pandas                
* pytorch               
* sklearn               
* matplotlib            
* math                  
* shutil                

Dataset: General aviation trajectory dataset (https://theairlab.org/trajair/)

Model framework: DiffTraj (https://github.com/Yasoz/DiffTraj)

### Generated result

![Validation set generation result](https://github.com/bonjer-26/Synthetic-spatial-data-generation/blob/main/traj_val.png?raw=true)

![Test set generation result](https://github.com/bonjer-26/Synthetic-spatial-data-generation/blob/main/traj_test.png?raw=true)

### Authors

Andriyan Saputra 
Student ID: 47717031
[@bonjer-26](https://github.com/bonjer-26)
[medium](https://medium.com/@andriyan-saputra78)

### Acknowledgments:
Zhu, Y., Ye, Y., Zhang, S., Zhao, X., James J.Q., (2023). DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model. Proceedings of the Advances in Neural Information Processing Systems, 36, 65168-65188. url: https://proceedings.neurips.cc/paper_files/paper/2023/file/cd9b4a28fb9eebe0430c3312a4898a41-Paper-Conference.pdf

Patrikar, J., Moon, B., Jean O., Scherer S. (2022). Predicting Like a Pilot: Dataset and Method to Predict Socially Aware Aircraft Trajectories in Non-Towered Terminal Airspace. 2022 International Conference on Robotics and Automation (ICRA), (pp. 2525-2531). doi: https://doi.org/10.1109/ICRA46639.2022.9811972







