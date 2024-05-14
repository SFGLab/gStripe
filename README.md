# gSTRIPE

Graph-based stripe caller using chromatin interaction data.

## Installation

1. _[Optional, but recommended]_ Set up a pyenv, conda or mamba python environment, and activate it.
2. Run `pip install git+https://github.com/SFGLab/gStripe.git` to install the gStripe package.

## Usage

Run `python -m -m gstripe.gstripe input_loops_file.bedpe output_directory` to call the stripes using loops from _input_loops_file.bedpe_ and place the results and statistics in _output_directory_.
This results in saving the candidate stripes to the output directory in a `.tsv` file. They should then be filtered by the user.

It is recommended to use the default values of all parameters specified in usage options (`python -m gstripe.gstripe --help`).

Use `--fix_bin_start` in case of binned data, where adjacent bins would overlap (i.e. when one anchor end is "15000" and the start of an adjacent anchor is also "15000"). Recommended for HiChIP.

In case of problems with multiprocessing, use `--max_workers=1`
