#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(/home/rgiordan/Documents/git_repos/trlib/build)
jupyter notebook
