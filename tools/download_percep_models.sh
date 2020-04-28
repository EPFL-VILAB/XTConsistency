##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/aXu4EFaznqtNzsE/download
unzip download
rm download
mv percep_models/* models/
rmdir percep_models