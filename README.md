# 🇨🇴 Observatorio Socioeconómico Departamental – Colombia

Aplicación interactiva desarrollada en **Streamlit** para el análisis socioeconómico a nivel departamental en Colombia.

## Características

- **35 indicadores** multidimensionales (laborales, macroeconómicos, sociales, seguridad, educación, salud e infraestructura).
- Periodo cubierto: **2010 – 2023**.
- Módulos incluidos:
  1. Exploración de Indicadores
  2. Análisis Exploratorio de Datos (EDA) + PCA
  3. Econometría de Series de Tiempo (VAR + Granger + IRF)
  4. Análisis Espacial (Ranking departamental + I de Moran)
  5. Machine Learning (Ridge, LASSO, SVR, MLP)
  6. Simulador de Política + Interpretabilidad SHAP

## Instalación

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador (normalmente en http://localhost:8501).

## Estructura del proyecto

```
observatorio-departamental/
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt
├── README.md
└── data/                   # (Opcional) Carpeta para futuros archivos Excel del DANE
```

## Notas importantes sobre los datos

- El panel actual es **sintético calibrado** (35 indicadores) porque la mayoría de series departamentales oficiales (desempleo, informalidad, IPM, pobreza monetaria, etc.) **no están disponibles de forma limpia y automática** vía API en datos.gov.co.
- Los únicos indicadores que se pueden obtener de forma relativamente automática por API son principalmente:
  - PIB departamental
  - Homicidios
  - Lesiones personales
  - Hurto a personas
  - Recursos de inversión pública (DNP)
- Para convertir el observatorio a **datos 100% oficiales**, se recomienda agregar una carpeta `data/` con los anexos Excel del DANE (GEIH departamental, IPM, pobreza monetaria, cuentas departamentales, etc.) y ampliar los cargadores.

## Requisitos del sistema

- Python 3.9 o superior
- 4 GB RAM mínimo (recomendado 8 GB)
- Conexión a internet solo para la primera instalación de paquetes

## Autor

Desarrollado para seminario de investigación – Observatorio Socioeconómico Departamental (Colombia).
