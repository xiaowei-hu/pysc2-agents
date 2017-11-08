<div align="center">
  <img src=images/MoveToBeacon.gif width="270px"/>
  <img src=images/CollectMineralShards.gif width="270px">
  <img src=images/DefeatRoaches.gif width="270px">
</div>


# PySC2 agents
This is a simple implementation of DeepMind's PySC2 RL agents. In this project, the agents are defined according to the original [paper](https://deepmind.com/documents/110/sc2le.pdf), which use all feature maps and structured information to predict both actions and arguments via an A3C algorithm.


## Requirements
- PySC2 is a learning environment of StarCraft II provided by DeepMind. It provides an interface for RL agents to interact with StarCraft II, getting observations and sending actions. You can follow the tutorial in [PySC2 repo](https://github.com/deepmind/pysc2) to install it.
```shell
pip install s2clientprotocol==1.1
pip install pysc2==1.1
```

- Python packages might miss: tensorflow and absl-py. If `pip` is set up on your system, it can be easily installed by running
```shell
pip install absl-py
pip install tensorflow-gpu
```


## Getting Started
Clone this repo:
```shell
git clone https://github.com/xhujoy/pysc2-agents
cd pysc2-agents
```


### Testing
- Download the pretrained model from [here](https://drive.google.com/open?id=0B6TLO16TqWxpUjRsWWdsSEU3dFE) and extract them to `./snapshot/`.

- Test the pretrained model:
```shell
python -m main --map=MoveToBeacon --training=False
```

- You will get the following results for different maps.

<table align="center">
  <tr>
    <td align="center"></td>
    <td align="center">MoveToBeacon</td>
    <td align="center">CollectMineralShards</td>
    <td align="center">DefeatRoaches</td>
  </tr>
  <tr>
    <td align="center">Mean Score</td>
    <td align="center">~25</td>
    <td align="center">~62</td>
    <td align="center">~87</td>
  </tr>
  <tr>
    <td align="center">Max Score</td>
    <td align="center">31</td>
    <td align="center">97</td>
    <td align="center">371</td>
  </tr>
</table>


### Training
Train a model by yourself:
```shell
python -m main --map=MoveToBeacon
```


### Notations
- Different from the original A3C algorithm, we replace the policy penalty term with epsilon greedy exploration.
- When train a model by yourself, you'd better to run several times and choose the best one. If you get better results than ours, it's grateful to share with us.


*Licensed under The MIT License.*
