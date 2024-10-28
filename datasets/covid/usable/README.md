# Usable

En esta carpeta están contenidas las imágenes de radiografías separadas en dos carpetas train y test. En ambas carpetas se encuentra un archivo .csv con las siguientes columnas:

- **filename** : Es el nombre del archivo que tiene la radiografía asociada al registro. Todas las imágenes se encuentran dentro de su correspondiente carpeta radiografías.
- **patient_id** : Id del paciente
- **class** : Diagnostico asociado a la radiografía. Posee unicamente dos valores *positive* y *negative*. Estas variable indica si el paciente desarrollo o presento la enfermedad COVID-10
- **data_source** : Es el nombre de la fuente/institución de la cual se consiguieron las imágenes asi como los diagnósticos
- **x1** : Valor normalizado de la componente horizontal que corresponde al primer punto(esquina superior izquierda) de la sección en que la red convolucional yolov7 segmento la clase 'pulmones'
- **y1** : Valor normalizado de la componente vertical que corresponde al primer punto de la sección en que la red convolucional yolov7 segmento la clase 'pulmones'
- **x2** : Valor normalizado de la componente horizontal que corresponde al segundo punto(esquina inferior derecha) de la sección en que la red convolucional yolov7 segmento la clase 'pulmones'
- **y2** : Valor normalizado de la componente vertical que corresponde al segundo punto de la sección en que la red convolucional yolov7 segmento la clase 'pulmones'
- **confidence** : Probabilidad que el modelo yolov7 esta seguro que en el rectángulo delimitado por los dos puntos se encuentra el objeto con categoría 'pulmones'