FROM python:3.8-slim-bullseye

EXPOSE 8000

# 设置当前目录为工作目录
WORKDIR /app

# Copy only the necessary files for pip install
COPY requirements.txt /app

# apt-get换源并安装依赖
RUN sed -i "s@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get update && apt-get install -y libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6
# 清理apt-get缓存
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# pip换源并安装python依赖
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r requirements.txt

# Copy the rest
COPY . /app
RUN pip3 install lib/paddlepaddle-2.4.2-cp38-cp38-linux_x86_64.whl

# CMD ["python3", "./main.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
