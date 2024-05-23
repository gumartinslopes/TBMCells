import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('../results/TBMCells/image4/test_3999_superpixels_2024-05-21 18:07:28.606554/overlay_result.png')

# Converter a imagem para o espaço de cores HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)

# Definir os intervalos de cor verde no espaço de cores HSV
limite_inferior = np.array([50, 100, 100])
limite_superior = np.array([70, 255, 255])

# Criar uma máscara para identificar pixels verdes na imagem
mascara_verde = cv2.inRange(imagem_hsv, limite_inferior, limite_superior)

# Encontrar contornos na máscara
contornos, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contar o número de objetos verdes
numero_de_objetos_verdes = len(contornos)

# Exibir o número de objetos verdes
print("Número de objetos com a cor verde (#00FF00):", numero_de_objetos_verdes)