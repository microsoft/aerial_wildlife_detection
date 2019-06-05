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

To start a server, run the "manage.py" script with the appropriate argument for the desired type of instance (i.e., labeling UI frontend or AI trainer).
Example: the following command would start a labeling UI frontend on the current machine, using the parameters specified in the `config` directory:
`python manage.py --instance=frontend`