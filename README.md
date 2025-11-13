# Proyecto de MLOps siguiendo la metodología CRISP-DM

Incluye empaquetado en PyPI, imagen en Docker Hub y pipelines de GitHub Actions para CI y CD.

---

## Repositorios y recursos

- Código fuente (GitHub):  
  <https://github.com/JaniMariQuesiRami/mlops-crispdm-ci-cd>

- Paquete en PyPI:  
  <https://pypi.org/project/mlops-crispdm-ci-cd/#history>

- Imagen en Docker Hub:  
  <https://hub.docker.com/repository/docker/jannisce2508/mlops-crispdm-ci-cd/general>

- Workflows de GitHub Actions:  
  <https://github.com/JaniMariQuesiRami/mlops-crispdm-ci-cd/actions>

---

## Instalación desde PyPI

En cualquier entorno con Python 3.9+:

~~~bash
pip install mlops-crispdm-ci-cd
~~~

---

## Uso básico del paquete

### Ejecutar la demo desde la terminal

El paquete expone un comando que recorre la “pipeline” CRISP-DM con prints de ejemplo:

~~~bash
mlops-crispdm-demo
~~~

Salida esperada:

~~~text
Demo pipeline CRISP-DM:
Business Understanding - demo
Data Understanding - demo
Data Preparation - demo
Modeling - demo
Evaluation - demo
Deployment - demo
~~~

### Uso como librería

Ejemplo mínimo:

~~~python
from mlops_crispdm import business, data_preparation

business.run()
data_preparation.run()
~~~

---

## Uso con Docker

La imagen en Docker Hub contiene el paquete instalado y ejecuta la demo por defecto.

### Descargar la imagen

~~~bash
docker pull jannisce2508/mlops-crispdm-ci-cd:latest
~~~

O una versión específica:

~~~bash
docker pull jannisce2508/mlops-crispdm-ci-cd:0.0.1
~~~

### Ejecutar el contenedor (se elimina al terminar)

~~~bash
docker run --rm --name mlops-crispdm-demo jannisce2508/mlops-crispdm-ci-cd:latest
~~~

---

## Estructura general del proyecto

Estructura simplificada:

~~~text
mlops-crispdm-ci-cd/
├── src/
│   └── mlops_crispdm/
│       ├── __init__.py
│       ├── business.py
│       ├── cli.py
│       ├── data_preparation.py
│       ├── data_understanding.py
│       ├── deployment.py
│       ├── evaluation.py
│       └── modeling.py.py
├── Dockerfile
├── pyproject.toml
└── .github/
    └── workflows/
        ├── ci.yml
        └── cd.yml
~~~

El código empaquetable está en `src/mlops_crispdm`.

---

## GitHub Actions: CI y CD

El proyecto usa dos workflows:

- `ci.yml` → Integración continua (CI).
- `cd.yml` → Despliegue continuo (CD) a PyPI y Docker Hub.

### Requisitos de configuración (secrets)

En GitHub, en `Settings → Secrets and variables → Actions`, se deben definir:

- `PYPI_API_TOKEN`  
  - API token de PyPI (scope para el proyecto o toda la cuenta).  
  - Valor con formato `pypi-...`.
- `DOCKERHUB_USERNAME`  
  - Usuario de Docker Hub.
- `DOCKERHUB_TOKEN`  
  - Access token de Docker Hub con permisos para publicar imágenes.

Con estos secrets, los workflows pueden publicar sin exponer credenciales en el código.

---

### CI: cómo funciona `ci.yml`

Archivo: `.github/workflows/ci.yml`

- Se activa en:
  - Cada `push` a la rama `main`.
  - Cada `pull_request` contra `main`.

Flujo:

1. Hace checkout del repo.
2. Instala Python 3.11.
3. Instala el paquete en modo editable (`pip install -e .`).
4. Ejecuta el comando `mlops-crispdm-demo`.

Si la instalación o la ejecución del comando fallan, el workflow marca error.  
Esto asegura que la “demo” mínima del proyecto siempre funciona antes de mergear cambios.

Más adelante se pueden añadir pruebas con `pytest` sobre el código que se vaya desarrollando.

---

### CD: cómo funciona `cd.yml`

Archivo: `.github/workflows/cd.yml`

- Se activa en:
  - Cualquier `push` de un tag que comience con `v` (por ejemplo `v0.0.2`).
  - Ejecución manual desde la pestaña “Actions” (`workflow_dispatch`).

Jobs principales:

1. `publish-pypi`
   - Construye el paquete (`python -m build`).
   - Sube las distribuciones a PyPI con `twine upload dist/*` usando `PYPI_API_TOKEN`.

2. `publish-docker`
   - Depende de que `publish-pypi` termine bien.
   - Hace login en Docker Hub con `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`.
   - Lee el tag Git creado (`v0.0.X`) y extrae la versión (`0.0.X`).
   - Construye y sube la imagen de Docker con dos tags:
     - `DOCKERHUB_USERNAME/mlops-crispdm-ci-cd:latest`
     - `DOCKERHUB_USERNAME/mlops-crispdm-ci-cd:0.0.X`

---

### Flujo de commits para activar el CD

Cada vez que se quiera sacar una nueva versión en PyPI y Docker Hub se debe:

1. Actualizar la versión en `pyproject.toml` y en `src/mlops_crispdm/__init__.py` (ejemplo: `0.0.1` → `0.0.2`).
2. Ejecutar:

~~~bash
git add .
git commit -m "chore: bump version to 0.0.X"
git tag v0.0.X
git push origin main
git push origin v0.0.X
~~~

Donde `X` es el número de versión que corresponda.  
El `push` del tag `v0.0.X` dispara automáticamente el workflow `cd.yml`.

---

## Conclusiones: escalabilidad del proyecto y próximos pasos

Este proyecto ya tiene una base de infraestructura lista para crecer:

- El empaquetado en PyPI permite reutilizar el código en otros proyectos o pipelines.  
- La imagen en Docker Hub facilita la ejecución uniforme en distintos entornos (local, servidor, nube).  
- Los workflows de GitHub Actions separan claramente CI y CD, lo que permite escalar el proceso de desarrollo sin cambiar la estructura principal.

Próximos pasos:

- Sustituir los `print` de demo por lógica real en cada etapa de CRISP-DM (carga de datos, preprocesamiento, entrenamiento, evaluación).  
- Agregar pruebas unitarias y de integración con `pytest` dentro de una carpeta `tests/` para mejorar la calidad del código.  
