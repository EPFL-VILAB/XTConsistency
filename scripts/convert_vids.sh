#!/bin/bash
mkdir -p ~/scaling/mount/simpsons_imgs

for f in ~/scaling/mount/shared/simpsons/*.mp4; 
do 
	b=$(basename "$f")
	echo $b
    ffmpeg -i "$f" -vf fps=5 ~/scaling/mount/simpsons_imgs/"$b"%d.png;  
done