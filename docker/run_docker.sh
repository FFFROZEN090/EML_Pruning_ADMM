user=$(whoami)
image_name=eml_final_project:v1
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user/EML_Pruning_ADMM/
workdir=/home/$user/EML_Pruning_ADMM/

docker run --gpus all -it -w $workdir -v $dir:$dir  --shm-size=40g --ulimit memlock=1 --ulimit stack=67108864 $image_name
