FROM python:3.8.16
COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
# CMD python app.py
CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app   