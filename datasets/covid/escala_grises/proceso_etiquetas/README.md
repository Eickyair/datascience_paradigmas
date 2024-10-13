# labelImg
Es un software bastante conocido para realizar de manera gráfica la etiquetación. En este caso buscamos generar un conjunto de datos para entrenar un modelo conocido como  [yolov7](https://github.com/WongKinYiu/yolov7) y segmentar las imágenes de radiografías para obtener el area donde se encuentran los pulmones.

## Instalación de labelImg

La guía oficial puede ser encontrada en [instalación](https://github.com/HumanSignal/labelImg) pero, en muchos casos hay errores durante su uso y al menos para la plataforma de <code>windows</code> puede usarse el <code>.exe</code> dentro de 
<code>plataforma/windows/labelImg.exe</code> o descargar directamente el comprimido [aqui](https://github.com/HumanSignal/labelImg/releases/tag/v1.8.1)

### Uso

1. Abrir el software
   1. ![Abrir Software](./assets/abrir_sf.png)
1. Activar algunas opcciones que facilitan el trabajo en la tab de view
   1. ![Tab view](./assets//activar_modos_etiquetas.png)
2. Configurar Folder en donde se encuentran las imagenes a etiquetar
   1. ![Directorio Imagenes](./assets//directorio_entrada_img.png)
3. Configurar Folder en donde se van a guardar las etiquetas
   1. ![Directorio Salida](./assets//directorio_salida.png)
4. Clickear para cambiar el formato de las etiquetas a YOLO
   1. ![Formato](./assets//formato_yolo.png)
5. Solo falta dar a Next Image y presionar el atajo w o el boton resaltado
   1. ![Etiquetas](./assets/crear_etiquetas.png)
6. Asegurarse de no crear mas etiquetas y asignar 'pulmones'
   1. ![Pulmones](./assets/asignar_et_pulmones.png)
7. Dar en ok, y continuar con cada imagen(en todo momento se puedo modificar esta imagen) presinando Next Image
   1. ![Imagen Etiquetada](./assets/resulatdo.png)