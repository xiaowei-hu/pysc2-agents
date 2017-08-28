# pysc2 agents

This is a simple implementation of DeepMind's pysc2 agents.  
Pysc2 is DeepMind's python component of the StarCraft II  
learning environment, which provides an interface for RL  
agents to interact with StarCraft II, getting observations  
and sending actions.  
We define the agents according to the original paper  
[StarCraft II: A New Challenge for Reinforcement Learning](  
https://deepmind.com/documents/110/sc2le.pdf). The agents use  
all feature maps and structured information to predict both  
actions and arguments via an A3C algorithm.


## Requirements
- pysc2
Pysc2 is a learning environment for StarCraft II provided  
by DeepMind. You can follow the tutorial in [pysc2 repo](  
https://github.com/deepmind/pysc2) to install it.

- tensorflow
- numpy
- gflag


## Getting Started
Clone this repo:
```bash
git clone https://github.com/xhujoy/pysc2-agents
cd pysc2-agents
```

### Testing
You may firstly download the pretrained model from  
[here](https://github.com/deepmind/pysc2) and extract  
them to `./snapshot/`.

Test the model:
```bash
python -m runRL --map=MoveToBeacon --training=False
```

You will get the following results for different  
MoveToBeacon/CollectMineralShards/DefeatRoaches maps.

<img src=images/MoveToBeacon.gif width="293px"/>  
<img src=images/CollectMineralShards.gif width="293px">  
<img src=images/DefeatRoaches.gif width="293px">

### Training

Train a model:
```bash
python -m runRL --map=MoveToBeacon
```

### Notations
- Different the original A3C algorithm, we replace the policy  
penalty term with \(\epsilon\)-greedy exploration.
- When train a model by yourself, you may try several times  
and choose the best model.


## Future Releases
- Improve stability and performance
- Full game supported
- Supervised learning supported
