"""
Redimensionar imágenes en escala de grises utilizando GPU.
"""
import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Redimensionar imágenes en escala de grises utilizando GPU.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directorio de entrada de las imágenes.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directorio de salida para las imágenes redimensionadas.')
    parser.add_argument('--width', type=int, default=256, help='Ancho de las imágenes redimensionadas.')
    parser.add_argument('--height', type=int, default=256, help='Altura de las imágenes redimensionadas.')
    args = parser.parse_args()
    return args

def resize_images(input_dir, output_dir, width, height):
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener lista de archivos de imagen en el directorio de entrada
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Configurar dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Utilizando dispositivo: {device}')

    # Definir transformación
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Resize((height, width)),  # Redimensionar
    ])

    # Procesar imágenes
    for image_file in tqdm(image_files, desc='Procesando imágenes'):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        try:
            # Cargar imagen en escala de grises
            image = Image.open(input_path).convert('L')
        except Exception as e:
            print(f'No se pudo abrir la imagen {input_path}: {e}')
            continue

        # Aplicar transformaciones
        with torch.no_grad():
            tensor = transform(image).unsqueeze(0).to(device)  # Añadir dimensión batch y mover a GPU
            tensor = tensor.squeeze(0)  # Quitar dimensión batch

        # Convertir tensor a imagen PIL
        resized_image = transforms.ToPILImage(mode='L')(tensor.cpu())

        # Guardar imagen redimensionada
        resized_image.save(output_path)

    print('Proceso completado.')

def main():
    args = parse_arguments()
    resize_images(args.input_dir, args.output_dir, args.width, args.height)

if __name__ == '__main__':
    main()
