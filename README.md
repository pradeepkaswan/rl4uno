# UnoBot
A reinforcement learning based agent trained to play the card game Uno (https://en.wikipedia.org/wiki/Uno_(card_game)). The project was implemented using Python with various modules for efficient arrays, machine learning and GUIs. Q-learning agents are trained inside the game environment and the resulting model can be analyzed inside a graphical version of the game including AI and human players, as well as a naive baseline algorithm.
## Training
To train the Q-model, simply start the train.py file with `python train.py` (to continue training an existing model, run `python train.py path/to/model.h5`). The model architecture and hyperparameters can be adjusted in the *agent.py* file. Further parameters regarding the Q-learning algorithm can be tuned inside the *train.py* file. Periodical model checkpoints (frequency adjustable in *agent.py*) will be saved under *models/\<timestamp>/model-\<epoch>.h5* and a tensorboard-compatible log file will be stored inside a *logs/\<timestamp>* folder.
## Playing
To run the game with a GUI, use `python play.py <player1> <player2> ...` and replace player arguments with either "AI", "Human" or "Naive". The AI tag will use the model specified inside the *play.py* file, adjust the model path variable to use a different model. If the AI player plays an illegal move, it will immediately be eliminated from the game. Selecting "Human" will allow the user to decide which moves to play in the game and the naive player will always select the first legal move inside the action space. At least two players have to be specified to start a game but player types can be mixed freely.
## Color selection
Players may now choose the next active colour after playing a Wild (type 13, “?”) or Draw-4 (type 14, “4+”) card:

- Human players are prompted via a colour-selection UI when they play a Wild or Draw-4 (configurable with `HUMAN_COLOUR_SELECTION` in `play.py`).
- AI players use a simple heuristic—picking the colour they have most of in hand—by default (`AI_COLOUR_SELECTION='heuristic'`), or can be set to pick randomly.

To revert to random colour assignment for either player type, set `HUMAN_COLOUR_SELECTION` or `AI_COLOUR_SELECTION` to `'random'`.

## Restarting the game
After a round ends, press **R** in the game window to start a new match with the same player types.
## Library requirements
- NumPy
- Keras
- tensorflow
- pygame
