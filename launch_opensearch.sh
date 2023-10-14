#pull opensearch version 2.10
OS_VERSION=2.10.0

docker pull opensearchproject/opensearch:$OS_VERSION

#start docker container 
docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:$OS_VERSION