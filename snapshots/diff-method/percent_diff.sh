#!/bin/bash

echo "scale=3; $(diff --side-by-side --suppress-common-lines $1 $2 | wc -l) / $(wc -l < $2)" | bc
