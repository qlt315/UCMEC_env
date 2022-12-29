# User-Centric Mobile Edge Computing Environment
## A single/multi-agent environment for Reinforcement Learning


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Please Cite](#please-cite)
* [Contributing](#contributing)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

This environment is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a world and offload computing tasks to the edge computing-enabled CPU via user-centric wireless transmission.







<!-- GETTING STARTED -->
## Getting Started

### Installation

Install using pip
```sh
pip install UCMEC
```
Or to ensure that you have the latest version:
```sh
git clone https://github.com/qlt315/UCMEC_env.git
cd UCMEC_env
pip install -e .
```


<!-- USAGE EXAMPLES -->
## Usage

Create environments with the gym framework.
First import
```python
import UCMEC
```

Then create an environment:
```python
env = gym.make("MA_UCMEC_env") # for single-agent
```
or 
```python
env = gym.make("SA_UCMEC_env") # for multi-agent
```


Similarly to Gym, step() function in multi-agent environment is defined as
```python
obs, reward, done, info = env.step(actions)
```

Where obs, rewards, done and info are LISTS of N items (where N is the number of agents). The i'th element of each list should be assigned to the i'th agent. 
action space is a LIST of N*M numbers that should be executed in that step.(M is the number of actions for each agent, like offloading decision, power allocation, etc.) 



<!-- CITATION -->
# Please Cite
1. The paper that first uses this implementation of single-agent UCMEC environment:
```
@ARTICLE{ucmec2022,
    author={Qin, Langtian and Lu, Hancheng and Wu, Feng},
    journal={IEEE Communications Magazine},   
    title={When User-Centric Network Meets Mobile Edge Computing: Challenges and Optimization},   
    year={2022},  
    volume={},  
    number={},
    pages={1-7},  
    doi={10.1109/MCOM.006.2200283}
    }
```





<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact

Langtian Qin - qlt315@mail.ustc.edu.cn

Project Link: [https://github.com/qlt315/UCMEC_env](https://github.com/qlt315/UCMEC_env)

