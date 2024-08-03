FROM python:3.11-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git
 
COPY ./req.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
WORKDIR app/
 
COPY . /app/rag_app

 
CMD ["streamlit", "run", "rag_app/conve_fe.py", "--server.port", "8080"]