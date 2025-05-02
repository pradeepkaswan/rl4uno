# RL4UNO

A modular UNO game environment with reinforcement-learning agents, human interaction, and naive baselines.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Pygame](https://img.shields.io/badge/pygame-2.6.1-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Play UNO against: 1) a trained AI model, 2) human players via GUI, or 3) a naive baseline strategy.

## Features

- OpenAI Gym–style `UnoEnvironment` for reinforcement learning.
- Three player modes:
  - **AI**: loads a Keras H5 model for policy inference.
  - **Human**: click-to-play GUI with card and colour selection.
  - **Naive**: plays the first legal move as a baseline.
- Wild (`?`) and Draw-4 (`4+`) colour-picker UI for humans.
- AI colour-choice heuristic (pick the colour you hold most of) or random.
- In-game restart: press **R** to reset the match with the same players.

## Installation

```bash
git clone https://github.com/pradeepkaswan/rl4uno.git
cd rl4uno
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Play the game

```bash
python play.py [Player1] [Player2] ...
```

- Player types: `AI`, `Human`, `Naive` (at least two players required).
- Example: `python play.py Human AI Naive`

### Training an AI agent

```bash
python train.py [model_path.h5]
```

- Customize network in `agent.py`, hyperparameters in `train.py`.
- Models saved under `models/<timestamp>/`, logs in `logs/<timestamp>/`.

## Configuration

Edit `play.py` constants:

```python
MODEL_PATH = 'example_model.h5'  # AI model file
HUMAN_COLOUR_SELECTION = 'prompt'  # 'prompt' or 'random'
AI_COLOUR_SELECTION = 'heuristic'  # 'heuristic' or 'random'
```

## Controls

- **Click** cards to play or draw from the stack.
- **Colour prompt** appears for Wild/Draw-4 when HUMAN_COLOUR_SELECTION='prompt'.
- **R**: restart game with the same players.
- Close window or **Ctrl+C** to exit.

## Project Structure

```
. ├── agent.py         # RL network architecture
. ├── environment.py   # UnoEnvironment game logic
. ├── play.py          # Pygame GUI runner
. ├── train.py         # Training script
. ├── simulate.py      # CLI self-play simulation
. ├── renderer.py      # Pygame rendering helpers
. ├── requirements.txt
. └── README.md
```

## Contributing

1. Fork the repo and create a branch (`feat/YourFeature`).
2. Commit with descriptive messages using Conventional Commits.
3. Submit a pull request for review.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
