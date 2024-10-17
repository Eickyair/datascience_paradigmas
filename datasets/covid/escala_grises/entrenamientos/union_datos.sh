#!/bin/bash
#!/bin/bash

output_path=""
source_path=""

# Función para obtener nombres de archivos sin extensiones
obtener_nombres_sin_extension() {
    local carpeta=$1
    find "$carpeta" -type f | sed 's/.*\///' | sed 's/\.[^.]*$//' | sort | uniq
}

# Comparar archivos entre dos carpetas
comparar_archivos() {
    local carpeta1=$1
    local carpeta2=$2
    nombres1=$(obtener_nombres_sin_extension "$carpeta1")
    nombres2=$(obtener_nombres_sin_extension "$carpeta2")

    for nombre in $nombres1; do
        if ! echo "$nombres2" | grep -q "^$nombre$"; then
            echo "El archivo $nombre no está en $carpeta2"
        fi
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



