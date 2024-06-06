#!/bin/bash

for n in adam sgd; do
    for m in sepsis los death_time; do
        echo "$m $n"
    done
done | xargs -P 2 -n 2 -I {} bash -c 'read -r m n <<< "{}"; OUTCOME=$m OPTIM=$n python MB-TCN/train.py'
