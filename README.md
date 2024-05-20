# gStripe

Graph-based stripe caller that uses chromatin loop data.

## Requirements:

* Python 3.10
* Operating system: Linux *(Ubuntu 18.04 or higher, recommended)* or Windows 10 *(or higher)*.

## Installation

1. *Optional, but recommended*: Create and activate a python envrionment using [pyenv](https://github.com/pyenv/pyenv), [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html).
2. Install gStripe, depending on the chosen source:
    * Installing directly from github: run `pip install git+https://github.com/SFGLab/gStripe.git`.
    * Installing from a zip file: run `pip install gStripe-v1.0.zip` in the directory containing the downloaded file. The name of the `.zip` archive might differ, depending on the download source and version number.

To verify, that the installation proceeded correctly, you can do the following:
1. Run `python -m gstripe.gstripe --help`: you should see the help message. If not, check if the correct environment has been activated.
2. Run gStripe on an example file provided in `examples/basic_test.bedpe` in the github repository or in the `.zip` archive, using the following command: `python -m gstripe.gstripe basic_test.bedpe .`. The console output should end with `[INFO] main(0.06s): All done.` (timing may vary) and two new files should be created: `./basic_test.bedpe.gstripes_raw.tsv` (results) and `basic_test.bedpe.gstripe.log` (log).

## Usage

Run `python -m gstripe.gstripe input_loops_file.bedpe output_directory` to call the stripes using loops from `input_loops_file.bedpe` and place the results (`input_loops_file.bedpe.gstripes_raw.tsv` by default) and the log file in _output_directory_.
This results in saving the candidate stripes to the output directory in a `.tsv` file. They should then be filtered by the user.

It is recommended to use the default values of all parameters specified in usage options (`python -m gstripe.gstripe --help`).

Use `--fix_bin_start` in case of binned data, where adjacent bins would overlap (i.e. when one anchor end is "15000" and the start of an adjacent anchor is also "15000"). Recommended for HiChIP.

In case of problems with multiprocessing, use `--max_workers=1`
