# Pokemon Gold RL - Program & Training Execution Flowchart

This document provides comprehensive flowcharts showing how the Pokemon Gold reinforcement learning system executes, from program startup through training and environment interaction.

## Table of Contents
1. [Overall Architecture](#1-overall-architecture)
2. [Training Execution Flow](#2-training-execution-flow)
3. [Environment Step Cycle](#3-environment-step-cycle)
4. [Reward Calculation Flow](#4-reward-calculation-flow)
5. [PPO Training Loop](#5-ppo-training-loop)
6. [Watch/Viewer Flow](#6-watchviewer-flow)

---

## 1. Overall Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        A[train.py] --> B[train_pokemon_gold.py]
        C[watch.py] --> D[watch_agent.py]
    end

    subgraph "Configuration"
        E[config.py]
    end

    subgraph "Environment"
        F[pokemon_gold_env.py]
        F --> G[gold_env/ram_helpers.py]
        F --> H[gold_env/event_flags.py]
        F --> I[gold_env/savestate_utilities.py]
        F --> J[gold_env/exploration.py]
        F --> K[gold_env/rewards.py]
    end

    subgraph "External Libraries"
        L[PyBoy Emulator]
        M[Stable-Baselines3 PPO]
        N[PyTorch]
    end

    B --> E
    B --> F
    B --> M
    B --> N
    D --> E
    D --> F
    D --> M
    F --> L

    style A fill:#90EE90
    style C fill:#87CEEB
    style E fill:#FFD700
    style F fill:#FFA07A
```

---

## 2. Training Execution Flow

```mermaid
flowchart TD
    Start([Run train.py]) --> AddPath[Add src/ to Python Path]
    AddPath --> LaunchTrain[Launch train_pokemon_gold.py]

    LaunchTrain --> LoadConfig[Load config.py]
    LoadConfig --> SetThreads{TORCH_NUM_THREADS set?}
    SetThreads -->|Yes| SetEnvVars[Set OMP/MKL/OPENBLAS thread limits]
    SetThreads -->|No| SkipThreads[Use default threading]
    SetEnvVars --> ImportLibs[Import NumPy, PyTorch, SB3]
    SkipThreads --> ImportLibs

    ImportLibs --> ConfigureGPU{CUDA Available?}
    ConfigureGPU -->|Yes| EnableCUDA[Enable cuDNN benchmark, TF32]
    ConfigureGPU -->|No| UseCPU[Use CPU device]
    EnableCUDA --> CreateEnvs
    UseCPU --> CreateEnvs

    CreateEnvs[Create Parallel Environments] --> ParallelType{NUM_ENVS > 1 && USE_SUBPROC?}
    ParallelType -->|Yes| SubProc[SubprocVecEnv - True Parallel]
    ParallelType -->|No| Dummy[DummyVecEnv - Sequential]

    SubProc --> WrapEnvs[Add VecMonitor]
    Dummy --> WrapEnvs
    WrapEnvs --> NormalizeEnv[Add VecNormalize - norm_reward=True]
    NormalizeEnv --> StackFrames{FRAME_STACK > 0?}
    StackFrames -->|Yes| AddStack[Add VecFrameStack]
    StackFrames -->|No| CheckCheckpoint
    AddStack --> CheckCheckpoint

    CheckCheckpoint{Checkpoint Exists?} -->|Yes| LoadModel[Load PPO Model from Checkpoint]
    CheckCheckpoint -->|No| CreateModel[Create New PPO Model]

    LoadModel --> SetupCallback
    CreateModel --> SetupCallback[Setup CheckpointCallback]

    SetupCallback --> TrainingMode{Training Mode?}
    TrainingMode -->|Time-Boxed| TimeBox[Set end_time from OVERNIGHT_TRAIN_SECONDS]
    TrainingMode -->|Step-Based| StepBased[Use TOTAL_TRAINING_STEPS]

    TimeBox --> TimeBoxLoop{time.time < end_time?}
    TimeBoxLoop -->|Yes| TrainChunk[model.learn - TRAIN_CHUNK_STEPS]
    TimeBoxLoop -->|No| SaveFinal
    TrainChunk --> CheckTime{> 1 min remaining?}
    CheckTime -->|Yes| TimeBoxLoop
    CheckTime -->|No| SaveFinal

    StepBased --> TrainFull[model.learn - TOTAL_TRAINING_STEPS]
    TrainFull --> SaveFinal

    SaveFinal[Save Final Model] --> Evaluate[Run Evaluation Episodes]
    Evaluate --> End([Training Complete])

    style Start fill:#90EE90
    style End fill:#FF6B6B
    style CreateModel fill:#FFD700
    style LoadModel fill:#87CEEB
    style TrainChunk fill:#FFA07A
    style TrainFull fill:#FFA07A
```

---

## 3. Environment Step Cycle

```mermaid
flowchart TD
    Start([env.step - action]) --> StorePos[Store position before action]

    StorePos --> HandleAction[_handle_action_input - action]
    HandleAction --> ActionType{Action Type?}

    ActionType -->|NOOP| IdleFrames[Tick frame_skip frames]
    ActionType -->|Button| PressButton[Send press event]

    PressButton --> HoldButton[Tick input_hold_frames]
    HoldButton --> ReleaseButton[Send release event]
    ReleaseButton --> WaitRelease[Tick post_release_frames]

    IdleFrames --> GetObs
    WaitRelease --> GetObs[_get_obs - Get screen]

    GetObs --> GetFrame[Get RGBA frame from PyBoy]
    GetFrame --> ConvertGray[Convert to grayscale - RGB//3]
    ConvertGray --> Downsample[Downsample 2x - 72x80]
    Downsample --> CheckPos[Get position after action]

    CheckPos --> DetectCollision{Is movement action && pos unchanged?}
    DetectCollision -->|Yes| IncrementCollision[Increment collision_count]
    DetectCollision -->|No| ResetCollision[Reset collision_count to 0]

    IncrementCollision --> UpdateTracking
    ResetCollision --> UpdateTracking[Update exploration tracking]

    UpdateTracking --> AppendAction[Append action to action_history]
    AppendAction --> AppendPos[Append position to position_history]
    AppendPos --> IncrementStep[Increment step_count]

    IncrementStep --> VerifyPause{VERIFY_PAUSE_STATE?}
    VerifyPause -->|Yes| CheckTextbox[Check wTextboxFlags RAM]
    VerifyPause -->|No| CalcReward
    CheckTextbox --> TextboxActive{Textbox active && not paused?}
    TextboxActive -->|Yes| ForcePause[Force dialog_depth=1]
    TextboxActive -->|No| CalcReward
    ForcePause --> CalcReward

    CalcReward[_calculate_reward] --> UpdateEpisodeStep[Increment _steps]
    UpdateEpisodeStep --> CheckTermination

    CheckTermination{_steps >= max_steps?} -->|Yes| Truncated[truncated=True]
    CheckTermination -->|No| NotTruncated[truncated=False]

    Truncated --> BuildInfo
    NotTruncated --> BuildInfo[Build info dict]

    BuildInfo --> Return([Return: obs, reward, terminated, truncated, info])

    style Start fill:#90EE90
    style Return fill:#FF6B6B
    style CalcReward fill:#FFD700
    style DetectCollision fill:#FFA07A
```

---

## 4. Reward Calculation Flow

```mermaid
flowchart TD
    Start([_calculate_reward]) --> Init[reward = 0.0]

    Init --> GetPos[Get current position]
    GetPos --> CheckPause{_pause_penalties?}

    CheckPause -->|Yes - Dialog/Battle| SkipMovement[Skip movement rewards]
    CheckPause -->|No| CheckStep[Apply STEP_PENALTY]

    CheckStep --> CheckTile{New tile?}
    CheckTile -->|Yes| NewTileReward[+NEW_TILE_REWARD]
    CheckTile -->|No| CheckRepeated{Repeated tile?}
    CheckRepeated -->|Yes| RepeatedPenalty[-REPEATED_TILE_PENALTY]
    CheckRepeated -->|No| CheckCollision

    NewTileReward --> CheckCollision
    RepeatedPenalty --> CheckCollision{Collision detected?}

    CheckCollision -->|Yes| CalcCollisionPenalty[Calculate collision penalty<br/>with exponential escalation]
    CheckCollision -->|No| NoveltyBonus
    CalcCollisionPenalty --> ClipCollision[Clip to COLLISION_PENALTY_MIN]
    ClipCollision --> NoveltyBonus

    NoveltyBonus{NOVELTY_BONUS_ENABLED?} -->|Yes| CalcNovelty[_calculate_novelty_bonus]
    NoveltyBonus -->|No| StuckDetection
    CalcNovelty --> StuckDetection

    StuckDetection{STUCK_DETECTION_ENABLED?} -->|Yes| CheckStuck[_detect_stuck]
    StuckDetection -->|No| ActionDiversity
    CheckStuck --> IsStuck{Stuck detected?}
    IsStuck -->|Yes| StuckPenalty[-STUCK_PENALTY]
    IsStuck -->|No| ActionDiversity
    StuckPenalty --> ActionDiversity

    ActionDiversity{ACTION_DIVERSITY_ENABLED?} -->|Yes| CalcDiversity[_calculate_action_diversity]
    ActionDiversity -->|No| DistanceReward
    CalcDiversity --> DistanceReward

    DistanceReward{DISTANCE_REWARD_ENABLED?} -->|Yes| CalcDistance[_calculate_distance_reward]
    DistanceReward -->|No| OneShot
    CalcDistance --> OneShot

    SkipMovement --> OneShot[One-Shot Event Rewards]
    OneShot --> CheckBadges{New badge earned?}

    CheckBadges -->|Yes| BadgeReward[+BADGE_REWARDS array<br/>15-50 per badge]
    CheckBadges -->|No| CheckWorldTile
    BadgeReward --> CheckWorldTile

    CheckWorldTile{New world tile?} -->|Yes| WorldTileReward[+NEW_WORLD_TILE_REWARD]
    CheckWorldTile -->|No| CheckMapBank
    WorldTileReward --> CheckMapBank

    CheckMapBank{New map bank?} -->|Yes| MapBankReward[+NEW_MAP_BANK_REWARD]
    CheckMapBank -->|No| CheckMapNumber
    MapBankReward --> CheckMapNumber

    CheckMapNumber{New map number?} -->|Yes| MapNumberReward[+NEW_MAP_NUMBER_REWARD]
    CheckMapNumber -->|No| CheckPokedex
    MapNumberReward --> CheckPokedex

    CheckPokedex{New species seen?} -->|Yes| SeenReward[+NEW_SPECIES_SEEN_REWARD]
    CheckPokedex -->|No| CheckCaught
    SeenReward --> CheckCaught

    CheckCaught{New species caught?} -->|Yes| CaughtReward[+NEW_SPECIES_CAUGHT_REWARD]
    CheckCaught -->|No| CheckParty
    CaughtReward --> CheckParty

    CheckParty{Party count increased?} -->|Yes| PartyReward[+NEW_POKEMON_CAUGHT_REWARD]
    CheckParty -->|No| CheckLevels
    PartyReward --> CheckLevels

    CheckLevels{Total levels increased?} -->|Yes| LevelReward[+POKEMON_LEVEL_UP_REWARD]
    CheckLevels -->|No| CheckStats
    LevelReward --> CheckStats

    CheckStats{Stats increased?} -->|Yes| StatReward[+HP/Atk/Def * MULTIPLIER]
    CheckStats -->|No| CheckPlotFlags
    StatReward --> CheckPlotFlags

    CheckPlotFlags{Plot flags enabled?} -->|Yes| IterateFlags[Check each PLOT_FLAG_REWARDS]
    CheckPlotFlags -->|No| ClipReward
    IterateFlags --> FlagSet{Flag newly set?}
    FlagSet -->|Yes| PlotReward[+PLOT_FLAG_REWARDS value]
    FlagSet -->|No| ClipReward
    PlotReward --> ClipReward

    ClipReward[Clip reward to ±REWARD_CLIP_ABS] --> Return([Return reward])

    style Start fill:#90EE90
    style Return fill:#FF6B6B
    style BadgeReward fill:#FFD700
    style CheckPause fill:#FFA07A
    style OneShot fill:#87CEEB
```

---

## 5. PPO Training Loop

```mermaid
flowchart TD
    Start([model.learn - total_timesteps]) --> InitRollout[Initialize rollout buffer]

    InitRollout --> CollectLoop{Collected n_steps?}
    CollectLoop -->|No| StepEnvs[Step all parallel environments]

    StepEnvs --> GetObs[Get observations from envs]
    GetObs --> Forward[Forward pass through policy network]
    Forward --> SampleAction[Sample actions from policy]
    SampleAction --> ExecuteActions[Execute actions in envs]
    ExecuteActions --> GetRewards[Collect rewards & next obs]
    GetRewards --> StoreBuffer[Store in rollout buffer]
    StoreBuffer --> CollectLoop

    CollectLoop -->|Yes| ComputeAdvantages[Compute advantages - GAE]
    ComputeAdvantages --> ComputeReturns[Compute returns]
    ComputeReturns --> EpochLoop{Trained n_epochs?}

    EpochLoop -->|No| ShuffleData[Shuffle rollout data]
    ShuffleData --> MinibatchLoop{Processed all minibatches?}

    MinibatchLoop -->|No| GetBatch[Get batch_size samples]
    GetBatch --> ForwardBatch[Forward pass - policy & value]
    ForwardBatch --> CalcLosses[Calculate PPO losses]

    CalcLosses --> PolicyLoss[Policy loss - clipped objective]
    PolicyLoss --> ValueLoss[Value loss - MSE]
    ValueLoss --> EntropyBonus[Entropy bonus]
    EntropyBonus --> TotalLoss[Total loss = policy + value - entropy]

    TotalLoss --> Backprop[Backward pass]
    Backprop --> ClipGrad[Clip gradients]
    ClipGrad --> UpdateWeights[Optimizer step]
    UpdateWeights --> MinibatchLoop

    MinibatchLoop -->|Yes| EpochLoop
    EpochLoop -->|Yes| CheckCallback{Checkpoint due?}

    CheckCallback -->|Yes| SaveCheckpoint[Save model checkpoint]
    CheckCallback -->|No| CheckTotal
    SaveCheckpoint --> CheckTotal{Reached total_timesteps?}

    CheckTotal -->|No| InitRollout
    CheckTotal -->|Yes| End([Training Complete])

    style Start fill:#90EE90
    style End fill:#FF6B6B
    style CalcLosses fill:#FFD700
    style SaveCheckpoint fill:#87CEEB
```

---

## 6. Watch/Viewer Flow

```mermaid
flowchart TD
    Start([Run watch.py]) --> AddPath[Add src/ to Python Path]
    AddPath --> LaunchWatch[Launch watch_agent.py]

    LaunchWatch --> LoadConfig[Load config.py]
    LoadConfig --> FindModel{Latest model exists?}
    FindModel -->|No| Error[Error: No trained model found]
    FindModel -->|Yes| LoadModel[Load PPO model]

    LoadModel --> CreateEnv[Create single PokemonGoldEnv]
    CreateEnv --> EnableWindow[enable_window=True]
    EnableWindow --> InitGUI[Initialize viewer_gui]
    InitGUI --> ResetEnv[Reset environment]

    ResetEnv --> MainLoop{Quit pressed?}
    MainLoop -->|No| CheckPause{Paused?}

    CheckPause -->|Yes| WaitResume[Wait for resume/step input]
    CheckPause -->|No| GetObs[Get current observation]

    WaitResume --> CheckKey{Key pressed?}
    CheckKey -->|Space| TogglePause[Resume]
    CheckKey -->|S| SingleStep[Step one frame]
    CheckKey -->|Other| WaitResume
    TogglePause --> GetObs
    SingleStep --> PredictAction

    GetObs --> PredictAction[model.predict - obs]
    PredictAction --> StepEnv[env.step - action]
    StepEnv --> UpdateGUI[Update GUI display]

    UpdateGUI --> CheckAutoReload{AUTO_RELOAD_MODELS?}
    CheckAutoReload -->|Yes| CheckInterval{Check interval passed?}
    CheckAutoReload -->|No| CheckReset
    CheckInterval -->|Yes| CheckNewModel{Newer model exists?}
    CheckInterval -->|No| CheckReset
    CheckNewModel -->|Yes| ReloadModel[Reload latest model]
    CheckNewModel -->|No| CheckReset
    ReloadModel --> CheckReset

    CheckReset{Reset pressed?} -->|Yes| ResetEnv
    CheckReset -->|No| CheckFastForward

    CheckFastForward{Fast-forward toggle?} -->|Yes| ToggleFPS[Toggle fast_forward flag]
    CheckFastForward -->|No| MainLoop
    ToggleFPS --> MainLoop

    MainLoop -->|Yes| CloseEnv[Close environment]
    CloseEnv --> End([Viewer Closed])

    style Start fill:#90EE90
    style End fill:#FF6B6B
    style MainLoop fill:#FFD700
    style PredictAction fill:#FFA07A
```

---

## Environment Reset Flow (Detailed)

```mermaid
flowchart TD
    Start([env.reset]) --> CheckEmulator{PyBoy initialized?}
    CheckEmulator -->|No| InitEmulator[_init_emulator]
    CheckEmulator -->|Yes| RunWarmup

    InitEmulator --> CreatePyBoy[Create PyBoy instance]
    CreatePyBoy --> RegisterHooks[Register dialogue/battle hooks]
    RegisterHooks --> RunWarmup

    RunWarmup[Tick BOOT_WARMUP_FRAMES] --> LoadState{init_state_path exists?}
    LoadState -->|Yes| LoadSavestate[Load savestate from disk]
    LoadState -->|No| CheckRequired{require_init_state?}

    CheckRequired -->|Yes| ErrorState[Error: Savestate required]
    CheckRequired -->|No| UseBootState[Use current boot state]

    LoadSavestate --> RenderFrame
    UseBootState --> RenderFrame[Tick 1 frame - render=True]

    RenderFrame --> ResetTracking[Reset _steps = 0]
    ResetTracking --> InitLastXY[Store _last_xy, _last_money]
    InitLastXY --> InitVisited[Initialize _visited_tiles set]
    InitVisited --> InitMapTracking[Initialize _visited_map_banks/numbers]
    InitMapTracking --> InitParty[Initialize party tracking]
    InitParty --> InitBadges[Initialize badge count]
    InitBadges --> InitWorld[Initialize world tiles]
    InitWorld --> InitPlotFlags[Initialize plot flags]
    InitPlotFlags --> InitPokedex[Initialize Pokedex tracking]

    InitPokedex --> InitDebug{DEBUG_PLOT_FLAGS?}
    InitDebug -->|Yes| OpenLogFile[Open plot_flags_debug.log]
    InitDebug -->|No| InitCollision
    OpenLogFile --> LogInitial[Log initial flag values]
    LogInitial --> InitCollision

    InitCollision[Reset collision tracking] --> InitExploration[Initialize exploration tracking]
    InitExploration --> SpawnPos[Store _spawn_position]
    SpawnPos --> GetObs[_get_obs - Get initial observation]
    GetObs --> Return([Return: obs, info])

    style Start fill:#90EE90
    style Return fill:#FF6B6B
    style LoadSavestate fill:#87CEEB
    style InitEmulator fill:#FFD700
```

---

## Key Observations

### Training Performance Optimizations
1. **Threading Control**: Limits CPU threads via `TORCH_NUM_THREADS` to prevent oversubscription
2. **GPU Optimizations**: Enables cuDNN benchmark and TF32 for faster training on modern GPUs
3. **Parallel Environments**: Uses `SubprocVecEnv` for true parallelism across CPU cores
4. **Frame Stacking**: Optional VecFrameStack for temporal information without RNNs
5. **Reward Normalization**: VecNormalize with `norm_reward=True` to stabilize training

### Environment Design
1. **Observation Processing**: Downsamples 160x144 → 80x72 grayscale for efficiency
2. **Action Timing**: Holds buttons for `INPUT_HOLD_FRAMES` to ensure game registers input
3. **Collision Detection**: Tracks consecutive collisions with exponential penalty escalation
4. **Pause System**: Disables movement penalties during dialogues/battles via hooks
5. **Anti-Stuck Mechanisms**: Multiple systems to prevent local optima (novelty, diversity, distance)

### Reward System Hierarchy
1. **Badges**: 15-50 points (highest priority)
2. **Pokedex Progress**: 2-10 points per species
3. **Exploration**: 0.02-5.0 points depending on scope
4. **Pokemon Training**: 0.5 points per level
5. **Movement Penalties**: -0.00001 to -1.0 for collisions/repetition

### Checkpoint System
1. **Automatic Resume**: Finds latest checkpoint on startup
2. **Periodic Saves**: Every `CHECKPOINT_SAVE_FREQUENCY` steps
3. **Time-Boxed Training**: Optional time limit instead of step limit
4. **Chunked Training**: Trains in chunks for frequent checkpoint opportunities

---

## Files Reference

| File | Purpose |
|------|---------|
| [train.py](train.py) | Training launcher script |
| [watch.py](watch.py) | Viewer launcher script |
| [config.py](src/config.py) | All configuration and hyperparameters |
| [train_pokemon_gold.py](src/train_pokemon_gold.py) | Main training loop implementation |
| [pokemon_gold_env.py](src/pokemon_gold_env.py) | Gym environment implementation |
| [watch_agent.py](src/watch_agent.py) | Interactive viewer with GUI |
| [gold_env/rewards.py](src/gold_env/rewards.py) | Reward calculation logic |
| [gold_env/exploration.py](src/gold_env/exploration.py) | Anti-stuck mechanisms |
| [gold_env/ram_helpers.py](src/gold_env/ram_helpers.py) | RAM reading utilities |
| [gold_env/event_flags.py](src/gold_env/event_flags.py) | Event flag parsing |
| [gold_env/savestate_utilities.py](src/gold_env/savestate_utilities.py) | Savestate management |

---

## Additional Resources

- **TensorBoard Monitoring**: `tensorboard --logdir _logs`
- **PPO Algorithm**: [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- **PyBoy Emulator**: [PyBoy GitHub](https://github.com/Baekalfen/PyBoy)
- **Pokemon Gold RAM Map**: [DataCrystal Wiki](https://datacrystal.tcrf.net/wiki/Pokémon_Gold_and_Silver/RAM_map)
