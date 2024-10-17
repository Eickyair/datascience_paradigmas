#!/bin/bash
#!/bin/bash

output_path=""
source_path=""

validar_archivo() {
    local ruta_archivo_origen=$1
    local ruta_archivo_destino=$2
    local lineas_validas=0
    local archivo_temp=$(mktemp)

    while IFS= read -r linea; do
        # Validar el formato de la línea
        if ! [[ $linea =~ ^0\ [0-9]+\.[0-9]{6}\ [0-9]+\.[0-9]{6}\ [0-9]+\.[0-9]{6}\ [0-9]+\.[0-9]{6}$ ]]; then
            # Validar que el primer elemento sea 0
            primer_elemento=$(echo $linea | awk '{print $1}')
            if [ "$primer_elemento" -ne 0 ]; then
                echo "Correccion 0: $linea"
                linea="0 $(echo $linea | cut -d' ' -f2-)"
                ((lineas_validas++))
            fi
        else
            ((lineas_validas++))
        fi
        echo "$linea" >> "$archivo_temp"
    done < "$ruta_archivo_origen"

    # Mostrar mensaje si el archivo tiene más de una línea válida y parar la ejecución
    if [ "$lineas_validas" -gt 1 ]; then
        echo "El archivo $ruta_archivo_origen tiene más de una línea válida."
        rm "$archivo_temp"
        return 1
    fi

    # Si hay una única línea válida, copiar el archivo temporal al destino
    if [ "$lineas_validas" -eq 1 ]; then
        mv "$archivo_temp" "$ruta_archivo_destino"
        echo "El archivo ha sido copiado a $ruta_archivo_destino."
    else
        echo "No se encontraron líneas válidas en el archivo."
        rm "$archivo_temp"
    fi
}
# Función para obtener nombres de archivos sin extensiones
obtener_nombres_sin_extension() {
    local carpeta=$1
    find "$carpeta" -type f | sed 's/.*\///' | sed 's/\.[^.]*$//' | sort | uniq
}

# Comparar archivos entre dos carpetas
comparar_archivos() {
    local carpeta_images=$1
    local carpeta_etiquetas=$2
    nombres1=$(obtener_nombres_sin_extension "$carpeta_images")
    nombres2=$(obtener_nombres_sin_extension "$carpeta_etiquetas")
    mkdir -p $output_path/etiquetas
    for nombre in $nombres1; do
        if ! echo "$nombres2" | grep -q "^$nombre$"; then
            echo "El archivo $nombre no está en $carpeta2"
            continue
        fi
        local ruta_archivo="$carpeta_etiquetas/$nombre.txt"
        echo $ruta_archivo
        validar_archivo "$ruta_archivo" "$output_path/etiquetas/$nombre.txt"
    done
}

validarEtiquetas() {
    local sub_carpeta=$1
    local carpeta_images="img"
    local carpeta_etiquetas="etiquetas"
    comparar_archivos "$sub_carpeta/$carpeta_images" "$sub_carpeta/$carpeta_etiquetas"
}

# Manejar opciones
while getopts "p:s:" opt; do
  case $opt in
    p) output_path="$OPTARG"
    ;;
    s) source_path="$OPTARG"
    ;;
    \?) echo "Opción inválida -$OPTARG" >&2
        exit 1
    ;;
  esac
done

# Verificar que las opciones no estén vacías
if [ -z "$output_path" ] || [ -z "$source_path" ]; then
  echo "Uso: $0 -p <output_path> -s <source_path>"
  exit 1
fi

echo "Archivos de $source_path serán unidos y guardados en $output_path"
sub_carpetas=(
    "erick"
    "imanol"
    "magno"
)
for sub_carpeta in "${sub_carpetas[@]}"; do
    validarEtiquetas "$source_path$sub_carpeta"
done



