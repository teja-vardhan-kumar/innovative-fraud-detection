FROM python:3.8-slim

WORKDIR /app

COPY *.py *.pkl requirements.txt /app/

RUN pip install --upgrade pip setuptools &&\
    pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "app.py" ]