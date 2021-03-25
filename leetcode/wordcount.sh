#!/usr/bin/env bash

# to count the number of uniq words, to create vocabulary file
cat wordcount.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{print $2,$1}'
