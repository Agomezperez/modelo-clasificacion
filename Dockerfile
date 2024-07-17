FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

RUN pip install opencv-python


COPY . .

EXPOSE 5000

CMD ["python", "app.py"]

