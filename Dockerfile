FROM python:3.11-slim

WORKDIR /app

# Versión que vendrá del tag (build-arg)
ARG PACKAGE_VERSION
ENV PACKAGE_VERSION=${PACKAGE_VERSION}

# Actualizar pip e instalar tu paquete desde PyPI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlops-crispdm-ci-cd==${PACKAGE_VERSION}

# Ejecutable definido en tu pyproject (entry point)
ENTRYPOINT ["mlops-crispdm-demo"]
