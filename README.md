# Cropland Detection

### [Repo link](https://github.com/AxelBogos/IFT6390-Data-Challenge-2)

### [Comet.ml Experiment workspace](https://www.comet.ml/ift6390-datachallenge-2/data-challenge-2/view/klyObIXPQL8JyGWTd2mA7VI7J/panels)

## Authors

- [@AxelBogos](https://www.github.com/AxelBogos)
- [@PulkitMadan](https://github.com/PulkitMadan)

## Abstract

While food scarcity and crop supply are perhaps some of humanity’s oldest challenges, the ongoing pandemic-induced shortages and the prospect of a rapidly changing climate makes addressing these challenges in novel ways especially
relevant. As extreme weather events are expected to grow both in frequency and intensity (NOAA National Centers for Environmental Information 2021), leveraging satellite data and tools developed by the machine learning community in order to assess the state of agricultural land might
help to both measure the impact of events such as wildfires or flooding on cropland, track population displacement and ultimately improve food security. The CropHarvest dataset (Tseng et al. 2021) aims at lowering the barrier of entry for ML practitioners interested in the aforementioned chal-
lenges. In this document, the pre-processing of the datasets, the models explored, the results obtained and finally a discussion of future endeavours and a reflection on our results will be conducted.

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
