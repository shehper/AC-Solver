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

This repository accompanies the paper *"What Makes Math Problems Hard for Reinforcement Learning: A Case Study."* It includes an implementation of the AC Environment in Gymnasium, two classical search algorithms (BFS and Greedy Search), and a PPO agent that works within this environment. Additionally, the repository contains Jupyter notebooks for reproducing the analyses and figures presented in the paper.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.