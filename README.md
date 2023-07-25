# Jump-start Reinforcement Learning

Implementation of [Jump-Start Reinforcement
Learning](https://arxiv.org/abs/2204.02372) (JSRL) with [Stable
Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Installation

```bash
pip install jsrl
```

## Usage

See `examples/train_jsrl_curriculum.py` or ``examples/train_jsrl_random.py`` for
examples on how to train TD3 + JSRL on the `PointMaze-v3` environment.

## References

```bibtex
@inproceedings{jsrl2022arxiv,
    title={Jump-Start Reinforcement Learning},
    author={Ikechukwu Uchendu, Ted Xiao, Yao Lu, Banghua Zhu, Mengyuan Yan, Joséphine Simon, Matthew Bennice, Chuyuan Fu, Cong Ma, Jiantao Jiao, Sergey Levine, and Karol Hausman},
    booktitle={arXiv preprint arXiv:2204.02372},
    year={2022}
}
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
@software{gymnasium_robotics2023github,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.2.0},
  year = {2023},
}
```
