# Car Insurance - Wauplin

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Purpose

This repository contains the code, documentation and report for a job application. The Challenge needed to be completed within a week. The main tasks are :
- upload data from CSV file to SQL database, alongside with computed features.
- make the best possible model for predicting damage.
- write a function to encapsulate predictions for a pair of variables
- (optional) : use above function in a simple web interface

## Disclaimer

The company name is never mentioned in the code or in the documentation. An environment variable is used to make it appear on the dashboard but it is a purely cosmetic addition.

## Repository architecture

```
- app/
  - dashboards/                     <- contains all scripts to display the dashboard pages
  (- podatki/podatki.csv)           <- folder containing uncompressed data
  - doc/
  - src/                            <- key resources for the algorithms
  - utils/                          <- generic but useful methods
  - constants.py                    <- define all constants of the project
  - dashboard.py                    <- app script
- db/
  - init.sql                        <- init script for the MySQL docker container
- docker-compose.yml                <- Docker compose configuration file
- Dockerfile                        <- Docker configuration file for Python image
- prod.env                          <- Environment variables to be set by docker-compose
- requirements.txt
- setup.cfg                         <- black and flake8 config
```

## Getting started

### Environment variables

Some environment variables are needed for the module to be operational. Default values are set in the `prod.env` file.

### Run

Project is based on [Docker-Compose](https://docs.docker.com/compose/). A Python image containing a [Streamlit dashboard](https://www.streamlit.io/) is built alongside a MySQL image. Before launching the app, the dataset must be downloaded and saved in `app/podatki/podatki.csv` (see above). Then, launch the app using :

```
docker-compose up -d
```

The app should be running on `127.0.0.1:8501`. The MySQL database is open on port `3306`.

### Online app

A deployed version of the dashboard is also available online. The url to this dashboard must have been sent to you.
