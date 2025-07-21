# Proyecto Recomendador de Películas

## Requisitos

- Python 3 instalado  
- VS Code (opcional)  
- Conexión a internet para instalar paquetes

## Instalación y configuración

1. Clona o descarga el proyecto y abre la carpeta en VS Code o terminal.

2. Crea y activa entorno virtual:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate
```
3. Instala librerías necesarias:
pip install pandas scikit-learn surprise matplotlib flask flask-cors

4. Descarga el dataset Movielens pequeño desde:
https://grouplens.org/datasets/movielens/
Descomprime el zip en la carpeta del proyecto, asegurándote que ratings.csv y movies.csv estén accesibles

5. ejecutar backend(Flask)
En la terminal con entorno activado, ejecuta:
python app.py
El servidor debería arrancar en:
http://127.0.0.1:5000

6. Ejecutar frontend
Para evitar problemas CORS y que el navegador permita peticiones al backend:

Desde la carpeta del proyecto ejecuta servidor simple:
python -m http.server 8000
Abre navegador y entra a:
http://localhost:8000/index.html

Usa el formulario para introducir el ID de usuario (ejemplo: 1) y obtener recomendaciones.

7. Solución de problemas comunes
Error "failed to fetch":
Asegúrate que el backend está corriendo y que has instalado y activado CORS en app.py:
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

8. Error al importar librerías:
Revisa que estés en el entorno virtual correcto y que las hayas instalado.

9. Problemas con paths de archivos CSV:
Asegúrate que ratings.csv y movies.csv están en la ruta correcta relativa a app.py.

10. Notas finales

- Recuerda usar 2 terminales, una para app.py y otra para abrir el servidor frontend.

- El backend usa un modelo básico de recomendación con la librería surprise.

- El frontend es una página simple que consume la API Flask y muestra resultados.

