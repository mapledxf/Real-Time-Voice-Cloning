FROM ubuntu:18.04


#RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list \
		&& sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list \
		&& apt-get clean \
		&& apt-get update
RUN apt-get install -y --no-install-recommends ffmpeg ca-certificates libportaudio2 git vim curl python3.7 \
		&& ln -sf /usr/bin/python3.7 /usr/bin/python \
		&& apt-get install -y --no-install-recommends build-essential python3.7-dev python3-distutils \
#		&& rm -rf /var/lib/apt/lists/* \
		&& curl -fsSL -o- https://bootstrap.pypa.io/get-pip.py -k | python 

WORKDIR /usr/src/app

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
	&& pip3 config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
	&& pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && git clone https://github.com/mapledxf/Real-Time-Voice-Cloning.git \
	&& cd Real-Time-Voice-Cloning \
	&& pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

