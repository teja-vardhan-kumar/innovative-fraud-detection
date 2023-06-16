FROM python:3.8-slim

#RUN apk add --no-cache py3-numpy

WORKDIR /app

COPY *.py *.pkl requirements.txt /app/

RUN pip install --upgrade pip setuptools &&\
    pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "app.py" ]