FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends mesa-utils x11-apps git vim cmake  build-essential && rm -rf /var/lib/apt/lists/* 

RUN git config --global http.sslverify false
RUN git clone https://github.com/libigl/eigen.git
RUN cd eigen && mkdir build && cd build && cmake .. && make -j4 && make install

RUN apt-get update && apt-get install -y libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 && rm -rf /var/lib/apt/lists/*

RUN mkdir /code && cd /code && git clone https://github.com/RainerKuemmerle/g2o.git && cd ./g2o && mkdir build && cd build && cmake ../ && make -j2 && make install -j2

RUN ldconfig
ENTRYPOINT ["tail", "-f", "/dev/null"]

