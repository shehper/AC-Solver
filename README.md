# AC-Solver

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Initializing the AC Environment](#initializing-the-ac-environment)
  - [Solving the Environment with PPO](#solving-the-environment-with-ppo)
  - [Performing Classical Search](#performing-classical-search)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)



## Overview

This repository accompanies the paper *"What Makes Math Problems Hard for Reinforcement Learning: A Case Study."* It includes an implementation of the Andrews-Curtis (AC) Environment in Gymnasium, two classical search algorithms (BFS and Greedy Search), and a PPO agent that works within this environment. Additionally, the repository contains Jupyter notebooks for reproducing the analyses and figures presented in the paper.

## Andrews-Curtis Conjecture
Andrews-Curtis conjecture is a long-standing open problem in combinatorial group theory and low-dimensional topology. It states that every balanced presentation of the trivial group could be transformed into the trivial presentation using actions on relators: inverses, conjugation and concatenation. More precisely, given presentation of trivial group of form $<x_{1}, x_{2}, \ldots, x_{n} | r_{1}, r_{2}, \ldots, r_{n}>$ can be transformed into $<x_{1}, x_{2}, \ldots, x_{n} |x_{1}, x_{2}, \ldots, x_{n}>$ using following set of moves:
- inverse: changing some $r_{i}$ with $r_{i}^{-1}$,
- conjugation: changing some $r_{i}$ with $qr_{i}q^{-1}$ for some $q$,
- concatenation: changing some $r_{i}$ with $r_{i}r_{j}$.

Many counterexamples were proposed for this conjecture. Finding such trivializing sequences is notoriously hard to find. Characteristics of this problem make it a perfect setup for Reinforcement Learning.

## Abstract and paper

Using a long-standing conjecture from combinatorial group theory, we explore, from multiple angles, the challenges of finding rare instances carrying disproportionately high rewards. Based on lessons learned in the mathematical context defined by the Andrews--Curtis conjecture, we propose algorithmic improvements that can be relevant in other domains with ultra-sparse reward problems. Although our case study can be formulated as a game, its shortest winning sequences are potentially $10^6$ or $10^9$ times longer than those encountered in chess. In the process of our study, we demonstrate that one of the potential counterexamples due to Akbulut and Kirby, whose status escaped direct mathematical methods for 39 years, is stably AC-trivial.

[Read full paper on arxiv](https://arxiv.org/)
[TODO: add link to arxiv]

## Installation

To work with the AC Environment or build upon it, you can simply install the package using pip:

```bash
pip install ac_solver
```

If you wish to reproduce the plots and analyses in the paper, you will need to clone the repository locally. Here is the recommended process:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AC-Solver.git
   cd AC-Solver
   ```

2. Install the package locally using Poetry:

   ```bash
   poetry install
   ```

   Alternatively, you can install the package using pip:

   ```bash
   pip install .
   ```

## Usage

After installation, you can start using the environment and agents as follows:

### Initializing the AC Environment

```python
from ac_solver.envs.ac_env import ACEnv
acenv = ACEnv()
```

<!-- By default, this initializes AC Environment with the following presentation from Akbulut-Kirby series, 

$$
\langle x, y | x^2 = y^3 , x y x = y x y \rangle
$$ -->

### Performing Classical Search

Specify a presentation and perform greedy search:

```python
presentation = [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]
from ac_solver.search.greedy import greedy_search
greedy_search(presentation)
```

### Solving the Environment with PPO

```python
from ac_solver.agents.ppo import train_ppo
train_ppo()
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks that reproduce the figures and results discussed in the paper:

- **`Sections-1-3-and-4.ipynb`**: Reproduces figures in Sections 1, 3, and 4 of the paper.
- **`Section-5.ipynb`**: Reproduces figures in Section 5 of the paper.
- **`Stable-AK3.ipynb`**: Provides code demonstrating that AK(3) is a stably AC-trivial presentation, a major result of the paper.

To run these notebooks, you must clone the repository locally as described above.

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Run tests with `pytest` and ensure your code is formatted with `black`:

   ```bash
   poetry run pytest
   poetry run black .
   ```

4. Submit a pull request with a clear description of your changes.

## Contact info
[TODO]
## Citation
[TODO}
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
