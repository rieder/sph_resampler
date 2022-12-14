# Installing

First, create a Python virtual environment, and activate it:
```sh
python3 -m venv env
. env/bin/activate
```
Then, install the prerequisites:
```sh
pip install -r requirements.txt
```
Then, you can run the script (`python resample_sph.py gas-original.amuse (FACTOR)`) or the plotting script (use -h for help).

# Notes

The current version is very much initial work - it has hardcoded values in it, a fixed random seed, and it probably has many bugs.
Don't use it except for developing and testing.
