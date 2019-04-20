# PPO

## Overview

To get a brief overview of this code, please refer to [this blog](http://www.meltycriss.com/2018/10/09/source-ppo/).

## Requirements

- Python 3
- Pytorch 0.4.0
- Visdom 0.1.8.5
- baselines (provided in this group)
	- Tensorflow 1.7.0 (CPU version is sufficient)
- gym_foa (provided in this group)
	- OpenAI Gym 0.10.5
- `pip install -r requirements.txt`

## Training

### Basic usages

1. Start a *Visdom* server with `visdom`, it will serve `localhost:8097` by default.
    
    ```bash
    visdom -env_path logs/181214
    ```

2. Run the following command to start training:

	```bash
	./run.sh python main.py --env-name "pos_v-v0" --log-dir logs/181214
	```

### More configurations

- `--indep`: Switch learning paradigm from CLDE (Centralized Learning and Decentralized Executing) to independent updating.
- `--unordered`: Use the index-free network architecture.
- `--load-dir`: For breakpoint continuation.
- `--vis-env`: Specify a Visdom environment to avoid unwanted replacement  induced by duplicated titles.

For full supported configurations, please refer to `arguments.py`.

## Testing

```bash
./run.sh python enjoy.py --env-name "pos_v-v0" --load-dir "logs/181214/pos_v-v0/clde_ordered/seed1/model/model0.pt"
```

**Caution**: remember to set the `--indep` flag if the model is derived by the independent updating strategy during training.
