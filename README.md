# Eye Disease Diagnosis System

A FastAPI-based web application for diagnosing eye conditions from images, with integrated Snellen vision test and dry eye assessment.

## Features
- Image-based disease detection (Eyelid Disease, Cataracts, etc.)
- Snellen visual acuity test
- Dry eye syndrome assessment
- Dockerized for easy deployment

## Prerequisites
- Docker Engine ([Install Guide](https://docs.docker.com/engine/install/))
- Python 3.9+ (for development only)
- Git

## Quick Start (For End Users)

### Using Docker (Recommended)
```bash
# Pull the pre-built image (includes ML models)
docker pull yahale/optiscan:latest

# Run the application
docker run -p 8000:8000 yahale/optiscan

# Access the app at:
http://localhost:8000
```

## Quick Start (For Developers)
```bash
# Clone and prepare models:
git clone https://github.com/yahale/optiscan.git
cd optiscan

# Download ML models from google drive:
# Datasets are also found in this link (optional)
https://drive.google.com/drive/folders/10_n7GLdp4thFNeqN_nyRfj0gFP4tRt6C?usp=sharing

#Alternatively you can extract the models from the docker image using the below commands:
docker pull yahale/optiscan:latest

# Create a stopped container
docker create --name optiscan_temp yahale/optiscan:latest

# Copy all model files from the container's root directory
docker cp optiscan_temp:/dry_eye_rf_model_updated.pkl .
docker cp optiscan_temp:/my_model.keras .
docker cp optiscan_temp:/snellen_model.pkl .
docker cp optiscan_temp:/snellen_label_encoder.pkl .


# Build a new Docker image with your changes
docker build -t optiscan-api:latest .

# Run a container from the new image
docker run -p 8000:8000 optiscan-api:latest
```
