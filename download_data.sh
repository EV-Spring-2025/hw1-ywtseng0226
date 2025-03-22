#!/bin/bash

if [ ! -d "data" ]; then
    gdown 1uWKDBGceL6XlP4Rl7zyntR53JrtgakDs -O data.zip
    unzip data.zip
fi
