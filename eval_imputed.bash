#!/bin/bash
export TESTING=true
for p in adam sgd; do
    for n in 3 6 12 18 24 32 40 48; do
        for m in sepsis los death_time; do
            echo "$p $m $n"
        done | xargs -P 3 -n 3 -I {} bash -c 'read -r p m n <<< "{}"; OPTIM=$p OUTCOME=$m HOUR_CAP=$n python MB-TCN/test.py'
    done
done