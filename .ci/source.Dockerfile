FROM aotriton:base

ARG GIT_HTTPS_ORIGIN GIT_NAME
RUN mkdir /root/build
WORKDIR /root/build

RUN git clone --recursive ${GIT_HTTPS_ORIGIN} -b ${GIT_NAME}
