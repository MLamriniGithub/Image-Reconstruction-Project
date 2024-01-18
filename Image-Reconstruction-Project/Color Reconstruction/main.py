import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Fonction pour restaurer un canal de couleur en utilisant la décomposition SVD
def svd_restore_color_channel(image_channel, mask, rank):
    # Appliquer la décomposition SVD sur le canal de couleur
    U, s, Vt = np.linalg.svd(image_channel, full_matrices=False)
    
    # Tronquer les matrices U, s et Vt pour obtenir une approximation de rang faible
    U_k = U[:, :rank]
    s_k = s[:rank] #sigma 
    Vt_k = Vt[:rank, :] #Vtransposé
    
    # Reconstruire l'image approximée à partir des matrices tronquées
    image_approx = U_k @ np.diag(s_k) @ Vt_k
    
    # Restaurer l'image en utilisant l'image approximée pour les pixels endommagés
    restored_image = np.where(mask, image_approx, image_channel)
    return restored_image

# Fonction pour restaurer une image en couleur en utilisant la décomposition SVD
def svd_restore_color(image, mask, rank):
    restored_image = np.zeros_like(image)
    # Restaurer chaque canal de couleur séparément
    for channel in range(3):
        restored_image[:, :, channel] = svd_restore_color_channel(image[:, :, channel], mask, rank)
    return restored_image

# Charger l'image en couleur et le masque
color_image_file_path = "cheval-abime2.png"
mask_file_path = "cheval-abime2-mask.png"
color_image = imread(color_image_file_path)
mask = imread(mask_file_path)

# Convertir le masque en un masque binaire
mask = (mask > 0)

# Choisir un rang pour l'approximation
rank = 50

# Restaurer l'image en couleur
color_image_restored = svd_restore_color(color_image, mask, rank)

# Afficher les images avant et après restauration
plt.figure()
plt.imshow(color_image)
plt.title("Original Color Image")

plt.figure()
plt.imshow(color_image_restored)
plt.title("Restored Color Image")

plt.show()
