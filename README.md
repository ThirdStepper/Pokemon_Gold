# Pokemon Gold RL Training

reinforcement Learning agent for Pokemon Gold using PPO (Proximal Policy Optimization) and PyBoy emulator. it's very much a work in progress, dont expect much.

## Project Structure

```
Pokemon_Gold/
├── src/                    # Source code (all Python modules)
│   ├── config.py          # Configuration and hyperparameters
│   ├── pokemon_gold_env.py # Gymnasium environment
│   ├── train_pokemon_gold.py # Training script
│   ├── watch_agent.py     # Live viewer for trained agents
│   ├── viewer_gui.py      # GUI components
│   ├── gui_helpers.py     # GUI helper functions
│   ├── make_boot_state.py # Create initial savestate
│   └── sanity_check.py    # Environment validation
├── rom/                   # Pokemon Gold ROM file
├── states/                # Savestate files
├── _models/               # Trained model checkpoints
├── _logs/                 # TensorBoard logs
├── tests/                 # Test files
├── train.py               # Launcher for training (convenience)
├── watch.py               # Launcher for viewer (convenience)
├── sanity_check.py        # Launcher for sanity check (convenience)
└── requirements.txt       # Python dependencies
```

## Running Scripts

You have two options for running scripts:

### Option 1: From Project Root (Recommended)

Use the convenience launcher scripts:

```bash
# Run sanity check
python sanity_check.py

# Train the agent
python train.py

# Watch a trained agent
python watch.py
```

### Option 2: From src/ Directory

Navigate to the `src/` directory first:

```bash
cd src

# Run sanity check
python sanity_check.py

# Train the agent
python train_pokemon_gold.py

# Watch a trained agent
python watch_agent.py
```

## Setup
0. setup a python 3.12 venv

1. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. place your Pokemon Gold ROM in `rom/Pokemon_Gold.gbc`

3. download the pokegold.sym file from: https://github.com/pret/pokegold/tree/symbols

4. then switch to the master branch and download the eventflags.asm file: https://github.com/pret/pokegold/blob/master/constants/event_flags.asm

5. edit the paths in config.py to point to both of these files

6. create initial savestate (skips intro):
   ```bash
   cd src
   python make_boot_state.py
   ```
   play through the intro, then press Enter to save the state.

7. run sanity check:
   ```bash
   python sanity_check.py
   ```

## Training

Start training:
```bash
python train.py
```

The training will:
- Save checkpoints to `_models/` every 50,000 steps
- Log metrics to `_logs/` for TensorBoard
- Resume from latest checkpoint if interrupted

Monitor with TensorBoard:
```bash
tensorboard --logdir _logs
```

## Watching the Agent

While training is running (or after):
```bash
python watch.py
```

Controls:
- `Q` - Quit
- `R` - Reset episode
- `SPACE` - Pause/Resume
- `F` - Fast forward toggle

## Configuration

Edit [src/config.py](src/config.py) to adjust:
- Training hyperparameters (learning rate, batch size, etc.)
- Reward weights (badges, exploration, Pokemon, etc.)
- File paths
- Episode length
- Frame skipping

## Notes

- All source code is in the `src/` directory
- Data directories (`_models/`, `_logs/`, `rom/`, `states/`) are in the root
- Launcher scripts in root are for convenience - they just add `src/` to Python's path
