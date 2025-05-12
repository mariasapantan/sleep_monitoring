FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY configs /app/configs/
COPY src /app/src/

COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
