# gSTRIPE

Graph-based stripe caller that uses chromatin loop data.

## Requirements:

* Python 3.10
* Operating system: Linux *(Ubuntu 18.04 or higher, recommended)* or Windows 10 *(or higher)*.

## Installation

1. *Optional, but recommended*: Create and activate a python envrionment using [pyenv](https://github.com/pyenv/pyenv), [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html).
2. Install gStripe, depending on the chosen source:
  * Installing directly from github: run `pip install git+https://github.com/SFGLab/gStripe.git`.
  * Installing from a zip file: run `pip install gstripe.zip` in the directory containing the downloaded file (The name of the file might differ, depending on the download source).

## Usage

Run `python -m -m gstripe.gstripe input_loops_file.bedpe output_directory` to call the stripes using loops from _input_loops_file.bedpe_ and place the results and statistics in _output_directory_.
This results in saving the candidate stripes to the output directory in a `.tsv` file. They should then be filtered by the user.

It is recommended to use the default values of all parameters specified in usage options (`python -m gstripe.gstripe --help`).

Use `--fix_bin_start` in case of binned data, where adjacent bins would overlap (i.e. when one anchor end is "15000" and the start of an adjacent anchor is also "15000"). Recommended for HiChIP.

In case of problems with multiprocessing, use `--max_workers=1`
