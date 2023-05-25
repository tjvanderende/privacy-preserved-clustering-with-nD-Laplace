#!/bin/sh


python main.py run-utility-experiments ../data/$2/$3.csv $1-piecewise $2
python main.py run-utility-experiments ../data/$2/$3.csv $1-laplace-truncated $2
python main.py run-utility-experiments ../data/$2/$3.csv $1-laplace $2
python main.py run-privacy-experiments ../data/$2/$3.csv $1-piecewise $2
python main.py run-privacy-experiments ../data/$2/$3.csv $1-laplace-truncated $2
python main.py run-privacy-experiments ../data/$2/$3.csv $1-laplace $2
python main.py run-comparison-experiment $4
