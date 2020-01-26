
sudo blobfuse ~/data/cameratraps --tmp-path=/mnt/resource/blobfusetmp  --config-file=fuse_cameratrap.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other
