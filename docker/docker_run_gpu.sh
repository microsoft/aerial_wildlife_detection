docker volume ls | grep -q aide_images || docker volume create aide_images
docker volume ls | grep -q aide_db_data || docker volume create aide_db_data

docker run --name aide_cnt \
 --gpus device=0 \
 --rm \
 -v `pwd`:/home/aide/app \
 -v aide_db_data:/var/lib/postgresql/10/main \
 -v aide_images:/home/aide/images \
 -p 8080:8080 \
 -h 'aide_app_host' \
 aide_app

 # Options:
 # --name   - container name
 # --gpus   - sets GPU configuration
 # --rm     - forces container removal on close (it doesn't affect volumes)
 # -v       - maps volume (note: aide_db_data and aide_images needs to be created before this script is executed)
 # -p       - maps ports
 # -h       - sets hostname (fixed hostname is required for some internal components)
