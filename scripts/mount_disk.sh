sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdd

rm -rf mount/data
sudo mkdir -p mount/data
sudo mkdir -p mount/data2
sudo mkdir -p mount/apolloscape
sudo mkdir -p mount/shared
sudo mount -o discard,defaults,ro,noload /dev/sdb mount/data
sudo mount -o discard,defaults,ro,noload /dev/sdc mount/data2

sudo mount -o discard,defaults /dev/sdd mount/apolloscape
sudo gcsfuse taskonomy-shared mount/shared

# sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdc
# rm -rf mount/data_full
# sudo mkdir -p mount/data_full
# sudo mount -o discard,defaults,rw /dev/sdc mount/data_full
