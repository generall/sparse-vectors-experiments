#!/bin/bash

DATASET=$1


wget "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${DATASET}.zip"

unzip "${DATASET}.zip"
