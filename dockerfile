#
# link: https://www.run.ai/guides/tensorflow/tensorflow-with-docker
#

FROM tensorflow/tensorflow:latest-gpu

# instalação necessaria para utilizar o openGL(open-cv)
RUN apt-get update && apt-get install -y mesa-utils libgl1-mesa-glx

# copia o dataset do github
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/Diegoomal/ebeer_dataset.git

COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
COPY app/ .
CMD [ "python", "main.py" ]