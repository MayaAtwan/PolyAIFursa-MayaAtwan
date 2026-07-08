FROM python:3.12-slim

WORKDIR /app

# Layer caching: copy requirements before source code.
# pip install only re-runs when requirements.txt changes, not on every commit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code comes last — changes on every commit but pip cache above is preserved
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
