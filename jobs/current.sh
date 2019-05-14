

job run --instance cloud4 --config "GAN_patches_3_blur6_subsetgan128_truebaseline_bestval" "python -m experiments.train_energy_subset consistency_paired_gaussianblur_truebaseline --batch-size 24 --use-patches --patch-size 3 --patch-sigma 0 --subset-size 128 --use-baseline --max-epochs 50" --shutdown



job run --instance cloud2 --config "GAN_patches_10_blur6_subsetgan128_10xupd_bestval" "python -m experiments.train_energy_subset consistency_paired_gaussianblur_subset --batch-size 24 --use-patches --patch-size 10 --patch-sigma 6 --subset-size 128 --max-epochs 50" --shutdown

job run --instance cloud6 --config "GAN_patches_10_blur6_subsetgan128_overfit" "python -m experiments.train_energy_subset consistency_paired_gaussianblur_subset --batch-size 24 --use-patches --patch-size 10 --patch-sigma 6 --subset-size 128 --max-epochs 50" --shutdown

job run --instance cloud7 --config "GAN_patches_10_blur6_subsetgan10000_overfit" "python -m experiments.train_energy_subset consistency_paired_gaussianblur_subset --batch-size 24 --use-patches --patch-size 10 --patch-sigma 6 --subset-size 10000 --max-epochs 50" --shutdown