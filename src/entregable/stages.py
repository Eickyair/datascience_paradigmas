from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
def loadImagen(path):
    return Image.open(path)
def resizeImagen(image, width, height):
    newImage = image.resize((width, height))
    newFileName = os.path.basename(image.filename)
    newFileName = ''.join(newFileName.split('.')[:-1]) + f'.jpg'
    newImage.filename = image.filename
    return newImage
def toGrayScale(image):
    newImage = image.convert('L')
    newImage.filename = image.filename
    return newImage
def compareImage(img, transformation):
    img_transformed = transformation(img)
    w,h = img.size
    w_transformed, h_transformed = img_transformed.size
    numKiloBytesOriginal = w*h*3/1024
    numKiloBytesTransformed = w_transformed*h_transformed*3/1024
    percentageReduction = 100 - (numKiloBytesTransformed/numKiloBytesOriginal)*100
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original', fontsize=20)
    ax[1].set_title('Transformed', fontsize=20)
    ax[1].imshow(img_transformed, cmap='gray')
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim([0, w])
        a.set_ylim([h, 0])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    ax[0].text(0, 0, f'Size: {numKiloBytesOriginal}[kB]', color='white', fontsize=16, verticalalignment='top', backgroundcolor='black')
    ax[1].text(400, 100, f'Size: {numKiloBytesTransformed:.2f}[kB]', color='black', fontsize=14, verticalalignment='top')
    ax[1].text(400, 200, f'Reduction: {percentageReduction:.2f}%', color='black', fontsize=14, verticalalignment='top')
    ax[0].set_xlabel(f'{w}x{h} pixels', fontsize=14)
    ax[1].set_xlabel(f'{w_transformed}x{h_transformed} pixels', fontsize=14)
    fig.tight_layout()
    plt.show()
    return img_transformed
def randomPaths(basePath,n=5):
    files = os.listdir(basePath)
    filter_images_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')]
    paths = [os.path.join(basePath, file) for file in filter_images_files]
    random_files = np.random.choice(paths, n)
    return random_files
def saveImages(images, dir_path = 'tmp_save_images',clean_dir=True):
    if clean_dir and os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            isDir = os.path.isdir(os.path.join(dir_path, file))
            if isDir:
                continue
            os.remove(os.path.join(dir_path, file))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    for i, img in enumerate(images):
        original_file = os.path.basename(img.filename)
        new_extension_jgp = original_file.replace('.png', '.jpg')
        img.save(os.path.join(dir_path, new_extension_jgp))
    absolute_path_dir_tmp = os.path.abspath(dir_path)
    return absolute_path_dir_tmp
def plot_images(dir_path,max_images=10,showTitle=True,k=1):
    files = os.listdir(dir_path)
    files_filter_images = [file for file in files if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')]
    num_pick = min(max_images, len(files_filter_images))
    random_images = files_filter_images[:num_pick]
    n_images = len(random_images)
    dimension = int(np.ceil(np.sqrt(n_images)))
    fig, ax = plt.subplots(dimension, dimension, figsize=(k*20,k*20))
    for i in range(n_images):
        img = Image.open(os.path.join(dir_path, random_images[i])).convert('L')
        ax[i//dimension, i%dimension].imshow(img, cmap='gray')
        if showTitle:
            ax[i//dimension, i%dimension].set_title(random_images[i], fontsize=20)
        ax[i//dimension, i%dimension].set_xticks([])
        ax[i//dimension, i%dimension].set_yticks([])
        ax[i//dimension, i%dimension].spines['top'].set_visible(False)
        ax[i//dimension, i%dimension].spines['right'].set_visible(False)
        ax[i//dimension, i%dimension].spines['bottom'].set_visible(False)
        ax[i//dimension, i%dimension].spines['left'].set_visible(False)
    for i in range(n_images, dimension*dimension):
        ax[i//dimension, i%dimension].axis('off')
    fig.tight_layout()
    plt.show()

def stage1(path_images,n):
    files = randomPaths(path_images,n)
    images = [loadImagen(file) for file in files]
    images_transformed = []
    for img in images:
        i_t = compareImage(img, lambda x: resizeImagen(x, 320, 320))
        images_transformed.append(i_t)
    print(len(images_transformed))
    abs_path = saveImages(images_transformed)
    return abs_path
import subprocess
def cnn(dir_images,useLastExp=False):
    if useLastExp:
        lastExp = os.listdir('./runs/detect')[0]
        lastExpPath = os.path.join('./runs/detect', lastExp)
        return lastExpPath
    tmp_dir = os.path.join(dir_images, "tmp")
    existTmpDir = os.path.isdir(tmp_dir)
    if existTmpDir:
        command = ["rm", "-r", tmp_dir]
        subprocess.run(command)
    os.makedirs(tmp_dir, exist_ok=True)
    for i, img in enumerate(os.listdir(dir_images)):
        isDir = os.path.isdir(os.path.join(dir_images, img))
        if isDir:
            continue
        isImage = img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')
        if not isImage:
            continue
        path_img = os.path.join(dir_images, img)
        img = loadImagen(path_img)
        filename = os.path.basename(path_img)
        filename_split = filename.split('.')
        file_name = '.'.join(filename_split[:-1]) + '.jpg'
        img = resizeImagen(img, 320, 320)
        new_file_path = os.path.join(tmp_dir, file_name)
        img.save(new_file_path)
    command = [
        "python", "../yolov7/detect2.py",
        "--weights", "../yolov7/best.pt",
        "--conf", "0.7",
        "--img-size", "320",
        "--save-txt",
        "--save-conf",
        "--source", tmp_dir,
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    lastExp = os.listdir('./runs/detect')[-1]
    lastExpPath = os.path.join('./runs/detect', lastExp)
    print(lastExpPath)
    return lastExpPath
def stage2(redimension_images_path,n):
    cnn_path = cnn(redimension_images_path)
    plot_images(cnn_path,n,showTitle=False,k=0.4)
import pandas as pd
def cargar_imagen(ruta_base, nombre_imagen):
    ruta_imagen = os.path.join(ruta_base, nombre_imagen)
    imagen = Image.open(ruta_imagen)
    return imagen


def get_subimage(image, bbox, size=None):
    """
    Extrae la subimagen de una imagen dada según un cuadro delimitador en formato [x, y, w, h],
    donde las coordenadas están normalizadas (valores entre 0 y 1).
    
    Parámetros:
    - image: Objeto de imagen PIL.
    - bbox: Lista o array con las coordenadas normalizadas [x, y, w, h],
            donde (x, y) es el centro del cuadro y (w, h) son el ancho y alto del cuadro.

    Retorna:
    - subimage: La subimagen recortada.
    """
    # Obtiene el tamaño de la imagen
    width, height = image.size
    
    # Desempaqueta los valores del cuadro delimitador
    x_center, y_center, box_width, box_height = bbox

    # Desnormaliza y convierte [x, y, w, h] a [x1, y1, x2, y2]
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)

    # Recorta la subimagen usando las coordenadas calculadas
    subimage = image.crop((x1, y1, x2, y2))
    if size is not None:
        subimage = subimage.resize(size)
    return subimage

def recortar_images_cnn(ruta_base,lista_nombres_img,predicciones,directorio_destino,size):
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino, exist_ok=True)
    if os.path.exists(directorio_destino):
        for file in os.listdir(directorio_destino):
            isDir = os.path.isdir(os.path.join(directorio_destino, file))
            if isDir:
                continue
            os.remove(os.path.join(directorio_destino, file))
    for i in range(len(lista_nombres_img)):
        imagen = cargar_imagen(ruta_base, lista_nombres_img[i])
        subimagen = get_subimage(imagen, predicciones[i], size)
        subimagen.save(os.path.join(directorio_destino, lista_nombres_img[i]))
        subimagen = subimagen.convert('L')
        subimagen.save(os.path.join(directorio_destino, lista_nombres_img[i]))
    return os.path.abspath(directorio_destino)
def stage3(detetcion_csv,ruta_imagenes,ruta_destino,n):
    df = pd.read_csv(detetcion_csv)
    path = recortar_images_cnn(ruta_imagenes,df['image'],df[['x1','y1','x2','y2']].values,ruta_destino,(224,224))
    plot_images(path,n,showTitle=False,k=0.4)




import pandas as pd
from sklearn.preprocessing import LabelEncoder
def visualize_image(image_path):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(image_path)
    plt.imshow(img, cmap='gray')
def pick_image(df, random_state=None, positive=None, n_images=1):
    df_sampled = df.sample(frac=1, random_state=random_state)
    if positive == 1:
        df_filtered = df_sampled[df_sampled['target'] == 1]
    elif positive == 0:
        df_filtered = df_sampled[df_sampled['target'] == 0]
    else:
        df_filtered = df_sampled
    images = df_filtered.head(n_images)
    return images['path'].tolist(), images['target'].tolist(), images.index.tolist()

def show_image(df, random_state=None, positive=None):
    import matplotlib.pyplot as plt
    image_path, target,_ = pick_image(df,random_state, positive)
    visualize_image(image_path[0])
    plt.title(f'target: {target[0]}')
def show_images(df, dimension=None, random_state=None, positive=None):
    if dimension is None:
        show_image(df, random_state, positive)
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        k = 2
        _, axs = plt.subplots(dimension[0], dimension[1],figsize=(k*dimension[1],k*dimension[0]))
        images_paths,targets,_ = pick_image(df, random_state, positive, dimension[0]*dimension[1])
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                if i*dimension[1]+j >= len(images_paths):
                    break
                img = mpimg.imread(images_paths[i*dimension[1]+j])
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].set_title(f'target: {targets[i*dimension[1]+j]}')
                axs[i, j].axis('off')
        plt.show()
def compare_images(df,dimensions=(2,2),random_state=None, split=False):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    if dimensions[1] % 2 != 0:
        raise ValueError('The second dimension must be an even number')
    total_images = dimensions[0]*dimensions[1]
    positive_images = total_images//2
    negative_images = total_images//2
    positive_paths, positive_targets,_ = pick_image(df, random_state, positive=1, n_images=positive_images)
    negative_paths, negative_targets,_ = pick_image(df, random_state, positive=0, n_images=negative_images)
    k=3
    fig, axs = plt.subplots(dimensions[0], dimensions[1], figsize=(k*dimensions[1],k*dimensions[0]))
    if split:
        for i in range(dimensions[0]):
            for j in range(dimensions[1] // 2):
                img = mpimg.imread(positive_paths[i * (dimensions[1] // 2) + j])
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].set_title(f'target: {positive_targets[i * (dimensions[1] // 2) + j]}')
                axs[i, j].axis('off')
            for j in range(dimensions[1] // 2, dimensions[1]):
                img = mpimg.imread(negative_paths[i * (dimensions[1] // 2) + (j - dimensions[1] // 2)])
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].set_title(f'target: {negative_targets[i * (dimensions[1] // 2) + (j - dimensions[1] // 2)]}')
                axs[i, j].axis('off')
    else:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if (i * dimensions[1] + j) % 2 == 0:
                    img = mpimg.imread(positive_paths[(i * dimensions[1] + j) // 2])
                    axs[i, j].imshow(img, cmap='gray')
                    axs[i, j].set_title(f'target: {positive_targets[(i * dimensions[1] + j) // 2]}')
                else:
                    img = mpimg.imread(negative_paths[(i * dimensions[1] + j) // 2])
                    axs[i, j].imshow(img, cmap='gray')
                    axs[i, j].set_title(f'target: {negative_targets[(i * dimensions[1] + j) // 2]}')
                axs[i, j].axis('off')
    plt.show()

def stage4(dimensiones=(9,7),random_state=1):

    df_radiografias = pd.read_csv('../../datasets/covid/usable/train/train.csv')
    df_radiografias['target'] = LabelEncoder().fit_transform(df_radiografias['class'])
    df_radiografias['path'] = df_radiografias['filename'].apply(lambda x: f'../../datasets/covid/usable/train/radiografias/{x}')
    show_images(df_radiografias,dimensiones,1)
    return df_radiografias
def make_dataset(df, random_state=None,n_images=1000):
    from PIL import Image
    import numpy as np
    print(df.shape)
    images_paths, targets, indexs = pick_image(df, random_state=random_state, n_images=n_images)
    images = []
    for image_path in images_paths:
        img = Image.open(image_path).convert('L')
        img = img.resize((224,224))
        img = np.array(img).flatten()
        images.append(img)
    images = np.array(images)
    targets = np.array(targets)
    return images, targets, indexs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def stage5(df_radiografias_train):
    X,y,indices = make_dataset(df_radiografias_train,random_state=40,n_images=1500)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=75))
    from sklearn.cluster import DBSCAN
    X_a = pipe.fit(X).transform(X)
    dbscan = DBSCAN(eps=155.55, min_samples=30, n_jobs=-1)
    dbscan.fit(X_a)
    df_radiografias_anomalias = df_radiografias_train.loc[indices]
    df_radiografias_anomalias['cluster'] = dbscan.labels_
    df_radiografias_anomalias = df_radiografias_anomalias[df_radiografias_anomalias['cluster'] == -1]
    anomalias = df_radiografias_anomalias.shape[0]
    n = int(np.floor(np.sqrt(anomalias)))
    if n % 2 != 0:
        n += 1
    show_images(df_radiografias_anomalias, dimension=(n,n))
def custom_predict_dbscan(dbscan,X_fit,eps,x):
    import numpy as np
    core_points = X_fit[dbscan.core_sample_indices_]
    for core_point in core_points:
        if np.linalg.norm(core_point-x) <= eps:
            return 0
    return -1
def stage6(df_radiografias):
    from sklearn.cluster import DBSCAN
    X,y,indices = make_dataset(df_radiografias,random_state=40,n_images=1500)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=75, random_state=40))
    dbscan = DBSCAN(eps=155.55, min_samples=22, n_jobs=-1)
    rutas_imagenes = df_radiografias['path']
    X_a = pipe.fit(X).transform(X)
    indices = rutas_imagenes.index
    indices_anmalias = []
    dbscan.fit(X_a)
    for ruta,indice in zip(rutas_imagenes,indices):
        img = Image.open(ruta).convert('L')
        img = img.resize((224,224))
        img = np.array(img).flatten()
        img = pipe.transform([img])
        label = custom_predict_dbscan(dbscan,X_a,155.55,img)
        if label == -1:
            indices_anmalias.append(indice)

    df_radiografias_anomalias = df_radiografias.loc[indices_anmalias]
    df_radiografias_sin_anomalias = df_radiografias.drop(indices_anmalias)
    n = int(np.floor(np.sqrt(len(indices_anmalias))))
    show_images(df_radiografias_anomalias, dimension=(n,n))
    return df_radiografias_sin_anomalias,df_radiografias_anomalias