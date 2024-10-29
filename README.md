See tokenised real space data in the folder named "domain data simulation". The crystal is using the Voronoi model.

See tokenised Fourier space data in the folder named "dp to tokens"

The transformer model is defined in the file "transformer_model.py". Run "main.py" for the training.

The current issue is: 
1. Query should be unbatched 2D or batched 3D tensor but received 4-D query tensor (Solved)
2. New model with normalisation and higher d_model and dimensions (Solved)
3. Improve the evaluation ()
