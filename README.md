# Raster Model

## Overview
Package for running optimal control on a raster based epidemic model.  Eventual aim is to optimise control spatially for a continuous landscape (e.g. Redwood Creek outbreak).  Here will test scalability of NLP solvers for this problem.

## Files
* **README.md** This document
* **raster_model.py** Primary module implementing the model
* **raster_model_fitting.py** module for fitting raster model to data
* **Ipopt/** Optimisation code. Uses Ipopt optimiser and must be built to generate executable
