To build image:
docker build -t superpoint:0.0.9 .

To run container:
docker run -v /home/ahab/data/testing_village_362x362/images:/usr/data --gpus all -tid superpoint:0.0.9

To run superpoint:
/usr/src/superpoint/build# ./superpoint --input /usr/data --model ../models/superpoint_v1.pt --device cuda

To build superpoint:
/usr/src/superpoint/build# rm -R *; cmake -DCMAKE_PREFIX_PATH=/usr/src/superpoint/libtorch ..; make
