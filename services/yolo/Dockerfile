FROM python:3.12-slim-trixie

WORKDIR /app

# System packages (rarely change — cached near top so rebuilds are fast)
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Layer caching: copy requirements before source code.
# pip install only re-runs when these files change, not on every code commit.
# Two separate RUN lines: if only requirements.txt changes, the torch layer stays cached.
COPY torch-requirements.txt requirements.txt ./
RUN pip install --no-cache-dir -r torch-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Source code comes last — changes on every commit but pip cache above is preserved
COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
