# Project to reproduce Paper id 32: Deep Patient Similarity Learning for Personalized Healthcare

CS598 Deep Learning for Healthcare Project

Link to the project Presentation
https://youtu.be/iNiYQhFDZzM

## Project Motivation

This paper aims at calculating a very important metric in Healthcare domain which is the Patient Similarity. Patient similarity is the concept of researching the most and least effective treatments based on the health records of like individuals with comparable health conditions. Adopting a patient similarity approach can enhance decision-making process in clinical practice, since there is a lot of clinical learning which measures the relative similarities between pairs of patients EHR data. 

### Built With
  * Python 3
  * Pytorch
  * Pandas
  * Numpy

## Data Location

http://www.emrbots.org/

100,000-patient (1.4GB) artificial EMR databases

## Folder structure
* `data` folder contains the raw files
* `results` folder contains the stats and results of the models
* **Root** folder contains all the code files

## Execution
Execute the `main.sh` script to run below steps
* Data Processing
* Baseline Modeling
* CNN Modeling
