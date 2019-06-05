# CV4Wildlife AL Service

Modular, multi-tier web framework for labeling image datasets with AI assistance and Active Learning support.

Main contributors: Amrita Gupta, Benjamin Kellenberger


## Overview

TODO


## Installation

TODO: steps:
1. Install required software (TODO: write setup.py file, eventually provide docker script?)
2. Provide parameters in `/config` directory with project-specific settings (database, interface parameters, etc.)
3. Launch script that sets up project (TODO: write bash script in projectSetup folder)
4. Populate database with data
5. Launch the desired instance (see below)


## Launching an instance

To start an instance, run the "runserver.py" script with the appropriate argument for the desired type of module (i.e., one of: "LabelUI", "AITrainer", "FileServer").
Example: the following command would start a labeling UI frontend on the current machine, using the parameters specified in the `config` directory:

`python runserver.py --instance=LabelUI`


It is also possible to run multiple modules on the same instance by providing comma-separated module names as an argument:

`python runserver.py --instance=LabelUI,AITrainer`