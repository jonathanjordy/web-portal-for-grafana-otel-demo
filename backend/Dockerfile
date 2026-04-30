FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cmdstan — overwrite=False skips if already installed
# The /root/.cmdstan directory is mounted as a volume in docker-compose
# so it persists across rebuilds and doesn't recompile every time
RUN python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=False, progress=False)"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]