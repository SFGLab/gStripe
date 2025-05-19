# gStripe

Graph-based architectural stripe caller.

## Requirements:

* Python >= 3.10
* Operating system: Linux *(Ubuntu 20.04 or higher, recommended)* or Windows 10 *(or higher)*.
* All the dependencies listed in `pyproject.toml`, notably `numpy >= 2.0.0` and `igraph >= 0.11.5`.

## Installation

1. *Optional, but recommended*: Create and activate a python envrionment using [pyenv](https://github.com/pyenv/pyenv), [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html).
2. Install gStripe, depending on the chosen source:
    * Install from PyPI (recommended): `pip3 install gstripe`
    * Installing directly from github: run `pip install git+https://github.com/SFGLab/gStripe.git`.
    
To verify, that the installation proceeded correctly, you can do the following:
1. Run `python -m gstripe.gstripe --help`: you should see the help message. If not, check if the correct environment has been activated.
2. Run gStripe on an example file provided in `examples/basic_test.bedpe` in the github repository or in the `.zip` archive, using the following command: `python -m gstripe.gstripe basic_test.bedpe .`. The console output should end with `[INFO] main(0.06s): All done.` (timing may vary) and two new files should be created: `./basic_test.bedpe.gstripes_raw.tsv` (results) and `basic_test.bedpe.gstripe.log` (log).

## Usage

The `gStripe` architectural stripe caller uses a discrete set of interactions (such as chromatin loops) to perform calling.

Run `python -m gstripe.gstripe input_interactions_file.bedpe output_directory` to call the stripes using interactions from `input_interactions_file.bedpe` and place the results (`input_loops_file.bedpe.gstripes_raw.tsv` by default) and the log file in _output_directory_.
This results in saving the candidate stripes to the output directory in a `.tsv` file. They should then be filtered by the user.

It is recommended to use the default values of all parameters specified in usage options (`python -m gstripe.gstripe --help`).

Use `--fix_bin_start` in case of binned data, where adjacent bins would overlap (i.e. when one anchor end is "15000" and the start of an adjacent anchor is also "15000"). Recommended for HiChIP.

In case of problems with multiprocessing, use `--max_workers=1`
