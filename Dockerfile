FROM python:3.11-slim

WORKDIR /app

# Copiamos solo lo necesario para construir el paquete
COPY pyproject.toml README.md ./
COPY src ./src

# Instalamos el paquete (como si viniera de PyPI, pero desde el código)
RUN pip install --no-cache-dir .

# Comando por defecto
ENTRYPOINT ["mlops-crispdm-demo"]
