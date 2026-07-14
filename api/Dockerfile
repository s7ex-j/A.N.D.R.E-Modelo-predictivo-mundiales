# Dockerfile — imagen lista para desplegar tu API en Render, Railway,
# Fly.io, un VPS, etc. Construye con:
#   docker build -t lambda26-api .
#   docker run -p 8000:8000 lambda26-api

FROM rocker/r-ver:4.4.1

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('plumber','data.table','xgboost','glmnet','nnet','jsonlite'), repos='https://cloud.r-project.org')"

WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["R", "-e", "pr <- plumber::plumb('plumber.R'); pr$run(host='0.0.0.0', port=8000)"]
