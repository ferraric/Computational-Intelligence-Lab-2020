#!/bin/sh
pipreqs --use-local --ignore pipreqs_ignored --savepath utilities/pipreqs.txt
sort utilities/non_imported_requirements.txt utilities/pipreqs.txt | uniq  > requirements.txt
cat requirements.txt
rm utilities/pipreqs.txt
