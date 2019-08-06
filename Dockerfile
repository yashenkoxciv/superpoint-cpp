FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL Name=superpoint Version=0.5.0

RUN apt-get -y update

# tzdata settings
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install GCC and opencv
RUN apt-get -y install gcc g++ cmake wget unzip libopencv-dev

# copy the stuff
COPY . /usr/src/superpoint
WORKDIR /usr/src/superpoint

# download and unpack LibTorch
RUN wget -nv --show-progress --progress=bar:force:noscroll https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-latest.zip
RUN unzip libtorch-shared-with-deps-latest.zip
RUN rm libtorch-shared-with-deps-latest.zip

# directory for data
RUN mkdir /usr/data

# compile the stuff
RUN mkdir build

WORKDIR /usr/src/superpoint/build
RUN cmake -DCMAKE_PREFIX_PATH=/usr/src/superpoint/libtorch ..
RUN make

# run the stuff
WORKDIR /usr/src/superpoint
CMD /usr/src/superpoint/build/superpoint --input /usr/data --model /usr/src/superpoint/models/superpoint_v1.pt --device cuda
