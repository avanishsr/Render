FROM python:3.11-slim

# Install libGL.so.1
RUN apt-get update && apt-get install -y libgl1