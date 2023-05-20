#!/bin/sh
python main.py run-utility-experiments ../data/seeds-dataset/rq1.csv 2d-pairwise seeds-dataset
python main.py run-utility-experiments ../data/seeds-dataset/rq1.csv 2d-laplace-truncated seeds-dataset
python main.py run-privacy-experiments ../data/seeds-dataset/rq1.csv 2d-pairwise seeds-dataset
python main.py run-privacy-experiments ../data/seeds-dataset/rq1.csv 2d-laplace-truncated seeds-dataset

python main.py run-compared-experiment RQ1

