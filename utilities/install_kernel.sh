#!/bin/bash

echo Installing Jupyter kernel named $1 with display name $2
ipython kernel install --name "$1" --user --display-name $2
