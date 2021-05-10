result=${PWD##*/}
res=${PWD}
docker kill $result
docker rm $result
sh build.sh
docker run --name formidrive_python -d -p 8080:8080 -v $(pwd):/var/www/html cadotinfo/opencv
