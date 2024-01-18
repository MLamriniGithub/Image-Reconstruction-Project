import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.pyplot as plt

def svd_restore(image, mask, rank):
    # Appliquer la décomposition en valeurs singulières (SVD)
    U, s, Vt = np.linalg.svd(image, full_matrices=False)

    # Tronquer les matrices U, s et Vt pour obtenir une approximation de rang faible
    U_k = U[:, :rank]
    s_k = s[:rank]
    Vt_k = Vt[:rank, :]

    # Reconstruire l'image approximée à partir des matrices tronquées
    image_approx = U_k @ np.diag(s_k) @ Vt_k

    # Restaurer l'image en utilisant l'image approximée pour les pixels endommagés
    restored_image = np.where(mask, image_approx, image)
    return restored_image

# Question 1: Charger l'image en noir et blanc et le masque
bw_image_file_path = "cheval-abime2-nb.png"
mask_file_path = "cheval-abime2-mask.png"
bw_image = imread(bw_image_file_path)
mask = imread(mask_file_path)

# Convertir le masque en un masque binaire
mask = (mask > 0)

# Restaurer l'image en noir et blanc
rank = 10  #  Ajustez cette valeur pour obtenir des résultats différents
bw_image_restored = svd_restore(bw_image, mask, rank)

# Calculer l'erreur quadratique et l'erreur selon la norme de Frobenius
error_quadratic = np.sum((bw_image - bw_image_restored) ** 2)
error_frobenius = np.linalg.norm(bw_image - bw_image_restored, 'fro')

print("Erreur quadratique:", error_quadratic)
print("Erreur selon la norme de Frobenius:", error_frobenius)

# Afficher les images
plt.figure()
plt.imshow(bw_image, cmap="gray")
plt.title("Original Black and White Image")

plt.figure()
plt.imshow(bw_image_restored, cmap="gray")
plt.title("Restored Black and White Image")

plt.show()
