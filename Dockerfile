FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-interaction

RUN conda install -c conda-forge r-base=4.3 r-ggplot2 -y
RUN R -e "install.packages(c('DESeq2', 'ANCOMBC', 'vegan', 'ggpubr'), repos='https://cloud.r-project.org')"

COPY . /app/

ENTRYPOINT ["python", "-m", "biomekit"]