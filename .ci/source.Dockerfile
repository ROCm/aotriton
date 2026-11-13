FROM aotriton:base

# The base image defaults to a non-root user; this image's RUNs need root, and the
# default goes back to that user at the end (see base.Dockerfile).
USER root

ARG GIT_HTTPS_ORIGIN GIT_NAME
RUN mkdir /root/build
WORKDIR /root/build

RUN git clone --recursive ${GIT_HTTPS_ORIGIN} -b ${GIT_NAME}

USER builder

# A build environment: nothing listens, so there is nothing to probe.
HEALTHCHECK NONE
