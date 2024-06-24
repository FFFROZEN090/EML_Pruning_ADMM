# Build docker image for the project in Mac OS

# Define the project name
PROJECT_NAME="eml_final_project"

# Build docker image
docker build -t $PROJECT_NAME:v1 -f Dockerfile .