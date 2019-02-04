FROM ib16/pytorch-cpu

WORKDIR /app
COPY ./code /app

ENTRYPOINT ["python", "main.py"]