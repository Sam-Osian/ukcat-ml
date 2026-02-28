# Classifying UK Charities

This repository is the home of the *[UK Charity Activity Tags](https://www.charityclassification.org.uk/)*, a project to classify
every UK registered charity using two classification taxonomies.

The project was a collaboration between [NCVO Research](https://www.ncvo.org.uk/policy-and-research), Dr Christopher Damm at the [Centre for Regional Economic and Social Research (Sheffield Hallam University)](https://www4.shu.ac.uk/research/cresr/staff/christopher-damm) and [David Kane](https://dkane.net/), an independent freelance researcher. The project was funded by Esmée Fairbairn Foundation.

The project started in Autumn 2020 with the first draft release of data in Summer 2021.

The classification and the results are licensed under a
[Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

## Using the python scripts

The scripts included in the repository were created using Python 3.9. They are likely to work with other versions of Python too.

### Installing dependencies

To use the python scripts, you'll need to install the required packages. The best way to do this is with a virtual environment:

```sh
python -m venv env  # creates a virtual environment in the ./env directory
# now activate the virtual environment
env\Scripts\activate  # (on windows)
source env/bin/activate   # (on unix/mac os)
pip -r requirements.txt  # installs the requirements 
```

You can then run the python scripts as described above - remember to activate the virtual environment every time you open a new terminal.

### Updating or adding additional dependencies

Dependencies are managed using [pip-tools](https://github.com/jazzband/pip-tools). First install it with:

```sh
python -m pip install pip-tools wheel setuptools
```

Then add any additional dependencies to `requirements.in`. Run `pip-compile` to create an updated `requirements.txt` file, and then run `pip-sync` to install the new requirements.

**!! Important - don't edit the `requirements.txt` file directly, it should only be edited with pip-compile**

## Repository contents

### `/data/`

Data outputs from the project. The following resources are available:

#### Classification schema

- ICNP/TSO: [`icnptso.csv`](data/icnptso.csv)
- UK-CAT: [`ukcat.csv`](data/ukcat.csv)

#### Manually classified charities

These files show the charities that were manually classified as part of this project. 

- [`sample.csv`](data/sample.csv)
- [`top2000.csv`](data/top2000.csv)

#### Categories for all charities

These files show the results of running automatic classification for UK-CAT and ICNP/TSO against
the latest lists of active and inactive charities in the UK.

The UK-CAT classification used a system of rules-based classification as described in the methodology. The ICNP/TSO classification uses a machine-learning model that is overwritten by any manual classifications found in the sample.

- [`charities_active-ukcat.csv`](data/charities_active-ukcat.csv)
- [`charities_inactive-ukcat.csv`](data/charities_inactive-icnptso.csv)
- [`charities_active-icnptso.csv`](data/charities_active-icnptso.csv)
- [`charities_inactive-icnptso.csv`](data/charities_inactive-icnptso.csv)


### `/docs`

This directory contains the project documentation, which is turned into a website using [mkdocs](https://www.mkdocs.org/).

You can run a local version of the docs using `mkdocs serve`.

The website is generated using Github actions.

### `/notebooks`

These notebooks contain code for processing the data, such as the machine learning model for ICNP/TSO classification.

To run the notebooks from with the virtual environment, use the following code ([from veekaybee.github.io](https://veekaybee.github.io/2020/02/18/running-jupyter-in-venv/)), after install the dependencies above

```sh
ipython kernel install --user --name=ukcat
jupyter notebook  # or `jupyter lab`
```

### `/ukcat`

This is a python module providing commands to fetch and apply the data from this project. 

The commands are as follows:

#### Fetch all charities

```sh
python -m ukcat fetch charities
```

This will create two CSV files containing data on charities. The files will be created in the `./data/` folder, and are `./data/charities_active.csv` and `./data/charities_inactive.csv`. 

These files are based on data from the [Charity Commission for England and Wales](https://register-of-charities.charitycommission.gov.uk/register/full-register-download), the [Scottish Charity Regulator](https://www.oscr.org.uk/about-charities/search-the-register/charity-register-download/) and the [Charity Commission for Northern Ireland](https://www.charitycommissionni.org.uk/charity-search/?pageNumber=1). Data is used under the Open Government Licence.

#### Fetch tags and sample

These are project internal scripts that fetch data from the airtable bases used by the project. They are used to create the two files `./data/sample.csv` and `./data/ukcat.csv` that are already available in the repo. They can only be operated correctly with the correct airtable credentials.

To fetch data you need to set two environment variables containing the airtable base ID and API key. The easiest way is to create a file called `.env` in this directory, and add the following lines (with the correct values):

```
AIRTABLE_API_KEY=keyGOESHERE
AIRTABLE_BASE_ID=appGOESHERE
```

An example can be found in `.env-sample`.

The commands to fetch the data are:

```sh
python -m ukcat fetch tags
python -m ukcat fetch sample
python -m ukcat fetch sample --table-name="Top charities" --save-location="./data/top2000.csv"
```

#### Create ICNP/TSO machine learning model

This script creates a Logistic Regression model for the ICNP/TSO categories, using the data found in
`sample.csv` and `top2000.csv`, based on the parameters created in `./notebooks/icnptso-machine-learning-test.ipynb`.

```sh
python -m ukcat train icnptso
```

By default this will save the model as a pickle file to `./data/icnptso_ml_model.pkl`.

#### Apply UK-CAT regex categories

This script uses the regular expressions from `./data/ukcat.csv` to apply tags to a list of charities.

```sh
python -m ukcat apply ukcat-regex --charity-csv "./data/charities_active.csv" -f "name" -f "activities"
python -m ukcat apply ukcat-regex --charity-csv "./data/charities_inactive.csv" -f "name" -f "objects"
```

This will create the `charities_active-ukcat-regex.csv` and `charities_inactive-ukcat-regex.csv` files by default. Each file gives a number of rows for each charity showing the UK-CAT tags that have been applied based on the regular expression keywords.

You can choose to include the name of the charity and the tag name by adding the `--add-names` option. You can also choose to add "parent" codes into the same data, by using the `--include-groups` option.

#### Train final UK-CAT OVR and hybrid models

After running the evaluation workflow, the selected best development configuration is stored in:

- `./data/ukcat_best_dev_config.json`

The final training commands read that file automatically and train on all labelled UK-CAT data by default:

```sh
python -m ukcat train ukcat-ovr --n-jobs 4
python -m ukcat train ukcat-hybrid --n-jobs 4
```

By default these save:

- OVR model artifact: `./data/ukcat_ml_ovr.pkl`
- Hybrid OVR model artifact: `./data/ukcat_ml_hybrid_ovr.pkl`
- Hybrid config: `./data/ukcat_ml_hybrid.json`

If `./data/ukcat_best_dev_config.json` does not exist, these commands will fail and tell you to run
`python -m ukcat evaluate dev-grid` first.

#### Apply trained UK-CAT OVR and hybrid models

Once the final artifacts have been trained, you can apply them to new charity files using the same row-per-code
output shape already used elsewhere in the repo:

```sh
python -m ukcat apply ukcat-ovr --charity-csv "./data/charities_active.csv"
python -m ukcat apply ukcat-hybrid --charity-csv "./data/charities_active.csv"
```

By default these save:

- `charities_active-ukcat-ovr.csv`
- `charities_active-ukcat-hybrid.csv`

You can also use the inactive file in the same way:

```sh
python -m ukcat apply ukcat-ovr --charity-csv "./data/charities_inactive.csv"
python -m ukcat apply ukcat-hybrid --charity-csv "./data/charities_inactive.csv"
```

As with the regex apply command, you can optionally add category names, sample the input rows for testing, and
apply manual overrides from the labelled files.

#### Apply ICNP/TSO categories

This script uses the machine learning model created in `./notebooks/icnptso-machine-learning-test.ipynb` to find the best ICNP/TSO category for a list of charities.

If the model doesn't already exist it will be created, using the files `sample.csv` and `top2000.csv`

```sh
python -m ukcat apply icnptso --charity-csv "./data/charities_active.csv" -f "name" -f "activities"
python -m ukcat apply icnptso --charity-csv "./data/charities_inactive.csv" -f "name" -f "objects"
```

This will create the `charities_active-icnptso.csv` and `charities_inactive-icnptso.csv` files that are included in the `./data/` folder. Each file gives a row per charity with the best estimated ICNP/TSO category, along with the model's estimated probability of the correctness of that category.

You can choose to include the name of the charity and the tag name by adding the `--add-names` option.

#### Evaluate UK-CAT comparison workflow

The evaluation CLI now focuses on one UK-CAT workflow: comparing the `regex`, `ovr`, and `hybrid`
approaches in a disciplined development and final-holdout process.

The workflow uses the labelled charities in `sample.csv` and `top2000.csv` as the canonical source data,
then creates a fixed development split and a fixed final holdout:

```sh
python -m ukcat evaluate make-split
```

This writes:

- `./data/ukcat_dev.csv`
- `./data/ukcat_final_test.csv`
- `./data/ukcat_holdout_split.json`

You can optionally override the fixed split defaults:

```sh
python -m ukcat evaluate make-split --random-state 2026 --final-test-size 0.2
```

To run the development-only grid search on the fixed dev split:

```sh
python -m ukcat evaluate dev-grid
```
Or...

```sh
python -m ukcat evaluate dev-grid --show-top 7 --n-jobs 4
```

This compares `regex`, `ovr`, and `hybrid`, prints summary metrics, and saves the winning development
configuration to:

- `./data/ukcat_best_dev_config.json`

To run the locked final holdout evaluation using that saved best configuration:

```sh
python -m ukcat evaluate final-holdout
python -m ukcat evaluate final-holdout --n-jobs 4
```

This trains the selected model settings on `./data/ukcat_dev.csv`, evaluates once on
`./data/ukcat_final_test.csv`, and prints final comparison metrics for `regex`, `ovr`, and `hybrid`.

To run both stages in sequence after the split already exists:

```sh
python -m ukcat evaluate run-full-workflow
python -m ukcat evaluate run-full-workflow --show-top 7 --n-jobs 4
```

This runs:

1. `dev-grid`
2. `final-holdout`

The evaluation CLI no longer exposes the older ICNP/TSO evaluation command, and the UK-CAT evaluation
surface is now fully comparison-oriented.


## Using the python scripts

The scripts included in the repository were created using Python 3.9. They are likely to work with other versions of Python too.

### Installing dependencies

To use the python scripts, you'll need to install the required packages. The best way to do this is with a virtual environment:

```sh
python -m venv env  # creates a virtual environment in the ./env directory
# now activate the virtual environment
env\Scripts\activate  # (on windows)
source env/bin/activate   # (on unix/mac os)
pip -r requirements.txt  # installs the requirements 
```

You can then run the python scripts as described above - remember to activate the virtual environment every time you open a new terminal.

### Updating or adding additional dependencies

Dependencies are managed using [pip-tools](https://github.com/jazzband/pip-tools). First install it with:

```sh
python -m pip install pip-tools wheel setuptools
```

Then add any additional dependencies to `requirements.in`. Run `pip-compile` to create an updated `requirements.txt` file, and then run `pip-sync` to install the new requirements.

**!! Important - don't edit the `requirements.txt` file directly, it should only be edited with pip-compile**
