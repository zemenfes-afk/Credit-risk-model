# Use a slim Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src $APP_HOME/src
# Copy transformers and model data (MUST be available for the API to load)
COPY data/processed $APP_HOME/data/processed

# Expose the port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]