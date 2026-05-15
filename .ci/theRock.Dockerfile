FROM aotriton:base

RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[libraries,devel]"

ENV PATH="/opt/venv/bin:${PATH}"
RUN echo 'source /opt/venv/bin/activate' >> ~/.bashrc
