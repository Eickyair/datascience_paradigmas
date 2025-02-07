# Proyecto Final Paradigmas

- Aguilar Martinez Erick Yair
- Martinez Muñoz Alan Magno
- Mendoza Saenz De Buruaga Imanol

## Objetivo

Resolver un problema de clasificación binaria para el diagnostico de la enfermedad COVID-19 mediante el uso de radiografías pulmonares

## Procesamiento

Los siguientes puntos resumen el preprocesamiento de los datos:

- Redimensionar imágenes a una misma resolución
- Pasar a escala de grises
- Resolver el problema de segmentación de pulmones
- Extraer los pulmones
- Redimensionar imágenes a nueva escala

## Filtrado y Transformación

- Mediante el uso de DBSCAN, pretender buscar imágenes anómalas para ser descartadas en la etapa de entrenamiento
- A traves PCA reducir la dimensionalidad permitiéndonos reducir el costo computacional para procesar las imágenes vectorizadas

## Optimización del Modelo

Una vez que obtengamos un modelo con los mejores resultados, hacer una búsqueda exhaustiva de hiperparametros que nos permita minimizar nuestro error de predicciones.

