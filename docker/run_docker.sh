user=$(whoami)
image_name=eml_final_project:v1
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user/EML_Pruning_ADMM/
workdir=/home/$user/EML_Pruning_ADMM/

docker run --gpus all -it -w /home/$USER/EML_Pruning_ADMM/ -v /home/$USER/EML_Pruning_ADMM/:/home/$USER/EML_Pruning_ADMM/ -p 8888:8888 --shm-size=40g --ulimit memlock=1 --ulimit stack=67108864 -e JUPYTER_ALLOW_INSECURE_WRITES=yes -e JUPYTER_TOKEN=2ded2de7654b09bb58029fbdb28ba41bef93050a628928a3 $image_name


