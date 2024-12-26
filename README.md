# SDoM - Stable Diffusion on Modal

This project was built to executes stable diffusion on Modal easly and persist the weights easily between different accounts.


## Getting Started

1. Generate a python local env with `python3 -m venv venv`
   1. Load the env with `source ./venv/bin/activate`
2. Install libraries executing `pip3 install -r requirements.txt`
3. Create a .env file using `cp .env.sample .env`
   1. Fill the environment variables


## Downloading Models

1. Fill the file `models.json` with desired models from available sources
2. Execute `modal run a1111_ui.py::download_models`


## Serving Frontend (temporally)

1. Execute `modal server a1111_ui.py`
