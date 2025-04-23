# Eye Disease Diagnosis System

A FastAPI-based web application for diagnosing eye conditions from images, with integrated Snellen vision test and dry eye assessment.

## Features
- Image-based disease detection (Glaucoma, Cataracts, etc.)
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
docker pull yourdockerhub/eye-diagnosis-app:latest

# Run the application
docker run -p 8000:8000 yourdockerhub/eye-diagnosis-app

Access the app at: http://localhost:8000


## Quick Start (For Developers)
