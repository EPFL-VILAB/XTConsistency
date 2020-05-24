##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/UG0ZlZXY47LKSaW/download
unzip download
rm download
cd data 
unzip -qqo albertville_rgb.zip
unzip -qqo albertville_normal.zip
unzip -qqo albertville_principal_curvature.zip
unzip -qqo almena_rgb.zip
unzip -qqo almena_normal.zip
unzip -qqo almena_principal_curvature.zip
rm albertville_rgb.zip albertville_normal.zip albertville_principal_curvature.zip almena_rgb.zip almena_normal.zip almena_principal_curvature.zip
cd -