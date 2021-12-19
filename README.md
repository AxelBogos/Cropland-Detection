
# Data Challenge 2

### [Repo](https://github.com/AxelBogos/IFT6390-Data-Challenge-2) for the 2nd data challenge of IFT6390. 
### [Comet.ml Experiment workspace](https://www.comet.ml/ift6390-datachallenge-2/data-challenge-2/view/klyObIXPQL8JyGWTd2mA7VI7J/panels)


## Authors

- [@AxelBogos](https://www.github.com/AxelBogos)
- [@PulkitMadan](https://github.com/PulkitMadan)


## Run Locally

Clone the project or unzip the submitted zip files

```bash
  git clone https://github.com/AxelBogos/IFT6390-Data-Challenge-2.git
```

Go to the project directory

```bash
  cd IFT6390-Data-Challenge-2
```

Create environment

```bash
 conda env create --file envname.yml
```

Activate the environment

```bash
  conda activate 6390 # Or any other env name of your choice
```
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`COMET_API_KEY` -> your [comet.ml](https://www.comet.ml/) API key. If you do not have one, set the boolean flag `use_comet` to false in any model execution.

## Running models
Each model is in a distinct directory in `./models`. All models (except `Neural Networks`) have functions `run_experiment` and `run_optimization`, which as their name indicate run a single experiment or the whole optimization. See the docstrings of their respective implementations for parameters.



