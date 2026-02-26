FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Set unbuffered output for logging
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Install required python packages
RUN pip install --no-cache-dir \
    pandas==1.5 \
    botorch==0.9.4 \
    gpytorch==1.11 \
    xgboost \
    scipy \
    tqdm \
    flask \
    gunicorn

# Create the directory structure
ENV PROGRAM_DIR=/opt/program
RUN mkdir -p $PROGRAM_DIR
WORKDIR $PROGRAM_DIR

# Copy the ALDE source code and Flask app
COPY src/ $PROGRAM_DIR/src/
COPY scripts/app.py $PROGRAM_DIR/app.py

# Add current directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:${PROGRAM_DIR}"

# Expose port 8080 for serving
EXPOSE 8080

# Serve with Gunicorn for production robustness
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "1800", "--graceful-timeout", "1800", "--workers", "1", "app:app"]
