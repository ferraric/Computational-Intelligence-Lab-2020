#!/bin/sh
pipreqs --ignore pipreqs_ignored,.mypy_cache --savepath utilities/pipreqs.txt
sort -u -t ' ' -k1,1 utilities/non_imported_requirements.txt utilities/pipreqs.txt | uniq > requirements.txt
rm utilities/pipreqs.txt
