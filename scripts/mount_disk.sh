sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
rm -rf mount/data
sudo mkdir -p mount/data
sudo mount -o discard,defaults /dev/sdb mount/data
sudo chmod a+w mount/data
