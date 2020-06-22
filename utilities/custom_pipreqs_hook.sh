#!/bin/sh
pipreqs --use-local --ignore pipreqs_ignored,.mypy_cache --savepath utilities/pipreqs.txt
sort utilities/non_imported_requirements.txt utilities/pipreqs.txt | uniq > temp_requirements.txt
tr -d '\015' < temp_requirements.txt > requirements.txt
rm utilities/pipreqs.txt
rm temp_requirements.txt
