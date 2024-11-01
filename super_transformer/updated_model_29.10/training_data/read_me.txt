The training datasets are, namely,
1. Seed information (x, y, amplitude, phase) (train_simulated__seeds.h5), x and y are between 0 and 16 for a 32 by 32 image; 
2. Full information of the diffraction pattern (DP) (x, y, intensity) (train_simulated_full_dp.h5); 
3. Partial information of the DP (amplitude below a threshold is removed) (train_simulated_thresholded_dp.h5); 
4. Local minima of the DP, later reformed by Gaussian functions (train_simulated_maxima_dp.h5). 

