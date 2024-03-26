FROM python:slim

RUN mkdir /app

COPY requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "flask", "run", "-h", "0.0.0.0", "--debug"]
