# DeepIceSheetDS2.5

**Improving DeepIceSheetDS2 using recurrent 2x downscaling**

## Method and model

There is no change in the main model, it's still a 3 layer SRCNN. But this time, **for dowscaling >2x**, the program is
recurrent. Meaning that, for downscaling 4x, the LR map will be first downscaled 2x with a 2x downscaled label, and then
the 2x downscaled prediction will be downcaled another 2x with the original label.

## Results

**4x dowscaling** (100000 epochs, learning rate 1e-6): bicubic error : 2.6e-4, model : 1.18e-4

**8x downscaling** (30000 epochs, learning rate 1e-4): bicubic error : 1.6e-2, model : 7.1e-4

**16x downscaling**

## Issues encountered 

Build this recurrent SRCNN using Tensorflow asked me more than 1 week of work, from understanding how setting outputs
from the first 2x dowscaling step as the input of the next step to reduce the computation complexity. Indeed the first 
runs of the model, the computation time was more than 20 seconds for 10 epochs, and I realized then that the more advanced 
epoch it was, the more time it will last. I understood that my algorithm was not optimized because I added new nodes to the 
graph at each epoch. These problem were solved by defining variables, models, layers outside of the training loop, with 
a massive use of dictionaries.   