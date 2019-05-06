# swingy-monkey-CZ
Clark Teeple, Zhuo Yang

## What is this code?
This code performs Q-learning on the game "Swingy Monkey". This is the 4th practical assignment for [CS181 at Harvard University](https://harvard-ml-courses.github.io/cs181-web/)

## How to run this code:
Choose your hyper parameters in `stub.py`.

 - Discretization
	 - **w_bin** - the number of bins for horizontal direction
	 - **h_bin** - the number of bins for the vertical direction
	 - **v_bin** - the number of bins for monkey velocity
 - **eta** - learning rate (0 < eta <1)
 - **gamma** - discount factor (0 < gamma <1)
 - **explore** - exploration rate (0 < epsilon <1)
 - **training_iters** - number of training iterations

Run `stub.py` to perform Q-learning on the game.

Run `displayResults.py` to calculate statistics and visualize the results.
