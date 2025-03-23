# CLS-Scientific_Computing-Assignment2
Solve the 2D wave equation for eigenmodes and eigenfrequencies of membranes (square, rectangle, circle) with fixed boundaries. Discretize, compute eigenvalues, analyze frequency dependence on size, and visualize time evolution. Also, solve the steady-state diffusion equation on a circular domain.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
- [File Descriptions](#file-descriptions)
- [Contributors](#contributors)
- [Git Fame](#git-fame)
- [License](#license)

## Description


## Getting Started

### Installation
First clone the repository.
```bash
git clone https://github.com/kingilsildor/CLS-Scientific_Computing-Assignment3
cd repository
```

### Prerequisites

To get the project running, install all the packages from the installer.
For this the following command can be used:
```bash
# Example
pip install -r requirements.txt
```

### Interface
Different modules can be run separately from their file.
But the main inferface for the project is `interface.ipynb` in the root folder.
This file uses all the functions that are important to run the code.

### Style Guide
For controbuting to this project it is important to know the style used in this document.
See the [STYLEGUIDE](STYLEGUIDE.md) file for details.


## File Descriptions

| File/Folder | Description |
|------------|-------------|
| `interface.ipynb` | Interface for all the code |
| `modules/config.py` | File for all constants |
| `modules/dla_algorithm.py` | File for simulation of DLA using growth probabilities based on concentration |
| `modules/gray_scott.py` | File for simulation of Gray-Scott model |
| `modules/grid.py` | File containing basic grid functions |
| `modules/random_walk_monte_carlo.py` | File for simulation of DLA using Monte Carlo random walk |
| `data/*` | Store for the data that the functions will write |
| `results/*`| Images and animations of the files |

## Contributors

List all contributors to the project.

- [Tycho Stam](https://github.com/kingilsildor)
- [Anezka Potesilova](https://github.com/anezkap)
- [Michael MacFarlane Glasow](https://github.com/mdmg01)

## Git Fame
Total commits: 61
Total ctimes: 494
Total files: 18
Total loc: 2235
| Author            |   loc |   coms |   fils |  distribution   |
|:------------------|------:|-------:|-------:|:----------------|
| kingilsildor      |  1350 |     28 |     11 | 60.4/45.9/61.1  |
| Anezka            |   705 |     21 |      4 | 31.5/34.4/22.2  |
| mdmg01            |   178 |      2 |      2 | 8.0/ 3.3/11.1   |
| Tycho Stam        |     2 |      6 |      1 | 0.1/ 9.8/ 5.6   |
| Anezka Potesilova |     0 |      4 |      0 | 0.0/ 6.6/ 0.0   |


Note: 
- Tycho Stam -> kingilsildor
- Michael MacFarlane Glasow -> mdmg01

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.