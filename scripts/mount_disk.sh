# sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
rm -rf mount/data
sudo mkdir -p mount/data
sudo mount -o discard,defaults,ro /dev/sdb mount/data
sudo gcsfuse taskonomy-shared mount/shared

# sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdc
# rm -rf mount/data_full
# sudo mkdir -p mount/data_full
# sudo mount -o discard,defaults,rw /dev/sdc mount/data_full