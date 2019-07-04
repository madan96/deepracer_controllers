#!/usr/bin/env bash

set -ex

export NODE_TYPE=SIMULATION_WORKER

python3 -m markov.run_pure_pursuit
