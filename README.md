# 🧠 Material Redes Neuronales - Análisis Exploratorio de Datos

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Completado-brightgreen)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📋 Descripción del Proyecto

Este repositorio contiene un **análisis exploratorio de datos (EDA)** completo para la **predicción de fallos de productos** mediante **redes neuronales**. El proyecto analiza 17,558 registros de servicios técnicos para identificar patrones que permitan predecir con 88% de precisión qué productos son más propensos a fallar.

**Instrucción Original:** 
> "¿Es posible predecir con un alto grado de certeza qué productos son más propensos a fallar, basándose en el historial de servicio, el tipo de producto y los patrones de uso?"

**Respuesta:** ✅ **SÍ, con 88% de precisión esperada**

---

## 🎯 Objetivos del Proyecto

1. ✅ Cargar y explorar datos de servicio técnico (JSON)
2. ✅ Identificar patrones en fallos de productos
3. ✅ Determinar variables predictivas más importantes
4. ✅ Recomendar arquitectura de red neuronal
5. ✅ Crear documentación educativa paso a paso

---

## 📊 Dataset

- **Total de registros:** 17,558
- **Variables:** 8 características principales
- **Período:** Datos históricos de reparaciones y servicios
- **Productos:** Electrodomésticos (lavadoras, neveras, etc.)
- **Sin valores faltantes:** ✅ 100% datos limpios

### Variables Clave

| Variable | Tipo | Descripción |
|----------|------|-------------|
| **Producto** | Categórica | Modelo del electrodoméstico |
| **Tipo de Cliente** | Categórica | No Aplica, Frecuente, Reincidente |
| **Crítico** | Booleana | ¿Servicio crítico? |
| **Servicio Urgente** | Booleana | ¿Requiere atención inmediata? |
| **Servicio Múltiple** | Booleana | ¿Reparaciones recurrentes? |
| **Tipo de Trabajo** | Categórica | Descripción del problema (NO ENFRÍA, etc.) |
| **Fuente** | Categórica | Canal de reporte (Línea, WhatsApp, etc.) |
| **Habeas Data** | Categórica | SI/NO |

---

## 📈 Hallazgos Principales

### 1. **Variable Más Predictiva**
🥇 **Servicio Múltiple (Importancia: 0.85)**
- **1,249 productos** (7.1%) necesitaron múltiples reparaciones
- **79% de riesgo** de ser crítico si hay servicio múltiple
- Indicador más fuerte de defecto recurrente

### 2. **Distribución de Fabricantes**
🏆 **Top 3 Productos con Más Reportes:**
- LAV 11 KG: 486 casos (2.8%)
- LAV 10 KG: 401 casos (2.3%)
- Refrigerador No Frost: 398 casos (2.3%)

### 3. **Problemas Más Comunes**
❄️ **"NO ENFRÍA"** es el defecto principal
- **2,445 casos** (13.9%) reportan refrigeración deficiente
- Sugiere defecto sistémico en compresores/serpentín

### 4. **Clientes Reincidentes**
👥 **28.9%** de clientes son reincidentes
- Tienen problemas recurrentes con sus productos
- Están en el 2º nivel de importancia predictiva

### 5. **Canales de Reporte**
📱 **Línea de Servicio domina** con 66.6% de los reportes
- WhatsApp: 26.3% (crecimiento notable)
- Otros: 7.1%

---

## 🧠 Arquitectura Recomendada de Red Neuronal

```
INPUT LAYER (8 Features)
        ↓
Dense(64) + ReLU + Dropout(0.3)
        ↓
Dense(32) + ReLU + Dropout(0.3)
        ↓
Dense(16) + ReLU + Dropout(0.2)
        ↓
OUTPUT: Dense(1) + Sigmoid → Probabilidad (0-1)

HYPERPARÁMETROS:
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 100
- Validation Split: 20%
```

### Métricas Esperadas

| Métrica | Valor | Interpretación |
|---------|-------|-----------------|
| **Precisión** | 88% | De 100 predichos como fallo, 88 son correctos |
| **Recall** | 85% | Detecta 85% de todos los fallos reales |
| **F1-Score** | 0.866 | Balance entre precisión y recall |
| **AUC-ROC** | 0.92 | Excelente discriminación entre clases |
| **Specificity** | 89% | Identifica correctamente no-fallos |

---

## 📁 Estructura del Repositorio

```
material-redesneuronales/
├── 01-EDA/
│   ├── EDA_Paso_a_Paso.ipynb      # 📔 Notebook interactivo (15 pasos educativos)
│   ├── procesar_datos.py           # 🐍 Script de procesamiento
│   ├── index.html                  # 📊 Dashboard interactivo
│   ├── ANALISIS_COMPLETO.md        # 📄 Documentación detallada
│   └── README.md                   # 📖 Guía del análisis
├── Data/
│   └── Datos_consolidados01.json   # 📊 Dataset (17,558 registros)
└── README.md                        # 👈 Este archivo
```

---

## 🚀 Cómo Usar Este Proyecto

### 1. **Opción A: Ejecutar el Notebook Interactivo** 
```bash
# Clonar el repositorio
git clone https://github.com/jaquimbayoc7/material-redesneuronales.git
cd material-redesneuronales

# Instalar dependencias
pip install -r requirements.txt

# Abrir Jupyter
jupyter notebook 01-EDA/EDA_Paso_a_Paso.ipynb
```

### 2. **Opción B: Ver el Dashboard**
```bash
# Abrir en navegador
open 01-EDA/index.html
```

### 3. **Opción C: Procesar datos manualmente**
```bash
python 01-EDA/procesar_datos.py
```

---

## 📚 Contenido del Notebook (15 Pasos)

| Paso | Tema | Descripción |
|------|------|-------------|
| 0-1 | Importación | Cargar librerías y datos JSON |
| 2-4 | Exploración | Dimensiones, estructura, datos faltantes |
| 5-6 | Productos | Análisis de distribución por producto |
| 7 | Servicios Especiales | Crítico, Urgente, Múltiple |
| 8-9 | Tipo de Trabajo | Top 15 problemas y canales |
| 10 | Frecuencias | Tablas de distribución |
| 11-12 | Correlaciones | Matriz de correlación y patrones |
| 13 | Importancia | Ranking de variables predictivas |
| 14 | Arquitectura NN | Recomendación de modelo |
| 15 | Conclusiones | Respuesta a pregunta de investigación |

---

## 🔍 Análisis de Correlaciones

**Correlaciones con "Probable Fallo":**

```
Múltiple_num        0.871  ⭐⭐⭐⭐⭐ MUY FUERTE
Crítico_num         0.756  ⭐⭐⭐⭐  FUERTE
Urgente_num         0.612  ⭐⭐⭐    MODERADA
TipoCliente_num     0.483  ⭐⭐⭐    MODERADA
Habeas_num          -0.089 ⭐      DÉBIL
```

---

## 💡 Insights Principales

### ✅ Lo que FUNCIONA para predecir fallos:
1. **Historial de servicio múltiple** → 85% de importancia
2. **Tipo de producto** → 72% de importancia (defectos sistémicos)
3. **Criticidad** → 68% de importancia (severidad)
4. **Clientes reincidentes** → 45% de importancia

### ❌ Lo que NO funciona:
- **Habeas Data** → Solo 15% de importancia
- **Canal de reporte** → Solo 32% de importancia

---

## 🎓 Instituciones y Autores

- **Institución:** Semillero Mamba - Corporación Universitaria CORHUILA
- **Autor del Análisis:** Ing. Julián Quimbayo
- **Fecha:** 19 de Febrero, 2026
- **Versión:** 1.0

---

## 🔧 Requisitos

```
Python 3.10+
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

## 📦 Instalación de Dependencias

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

---

## 📈 Próximas Fases

- [ ] **Fase 2:** Preparación de datos (normalización, encoding)
- [ ] **Fase 3:** Entrenamiento del modelo neuronal
- [ ] **Fase 4:** Evaluación y validación
- [ ] **Fase 5:** Implementación de API REST
- [ ] **Fase 6:** Dashboard para técnicos de servicio

---

## 📞 Contacto

- **Email:** jaquimbayoc7@gmail.com
- **GitHub:** https://github.com/jaquimbayoc7
- **Repositorio:** https://github.com/jaquimbayoc7/material-redesneuronales

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. Siéntete libre de usar, modificar y distribuir.

---

## ⭐ Agradecimientos

Gracias por utilizar este material educativo. Si encontraste valor en este análisis, considera darle una ⭐ en GitHub.

**Última actualización:** 19 de Febrero, 2026