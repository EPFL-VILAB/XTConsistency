##!/usr/bin/env bash

SCRIPT_DIR=$( dirname "$0" )

FILE=./models/rgb2normal_consistency.pth
if [ -f "$FILE" ]; then
    echo "Found consistency network $FILE: skipping download of these networks"
else
    echo "Downloading consistency networks..."
   sh $SCRIPT_DIR/download_models.sh
fi

FILE=./models/normal2curvature.pth
if [ -f "$FILE" ]; then
    echo "Found perceptual network $FILE: skipping download of these networks"
else
   echo "Downloading perceptual networks..."
   sh $SCRIPT_DIR/download_percep_models.sh
fi

FILE=./models/rgb2principal_curvature.pth
if [ -f "$FILE" ]; then
    echo "Found energy-graph specific network $FILE: skipping download of these networks"
else
    echo "RGB2X energy networks..."
    # Get energy-graph-specific links
    wget https://drive.switch.ch/index.php/s/aZDOEBixS4W7mBL/download
    unzip download
    rm download
    mv energy_graph_edges/* models/
    rmdir energy_graph_edges
fi

