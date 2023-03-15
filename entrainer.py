from model import create_model
from preprocessiong import preprocess_data

X, y = preprocess_data()

X_train, y_train = X[:970], y[:, :970] #X_train=on sélectionne les 970 premières images de captcha pour l'ensemble d'entraînement///
#y_train=on sélectionne les 970 premières étiquettes correspondantes (y_train) pour les cinq caractères du captcha.
#x_train donnée de l'entrainement, y_train= etiquette de l'entrainement

#, X[970:] récupère tous les exemples à partir de l'index 970 jusqu'à la fin de l'ensemble de données,
# tandis que y[:, 970:] récupère toutes les étiquettes correspondantes (chaînes de caractères de 5 caractères) à partir du 971ème index jusqu'à la fin.
X_test, y_test = X[970:], y[:, 970:]

model = create_model()
model.summary()


#batch_size=nombre d'exemples qui seront traités avant de mettre à jour les poids du réseau lors de la rétropropagation de l'erreur
# nombre d'itérations à effectuer sur l'ensemble des données d'entraînement
#epochs=nombre de fois où l'ensemble de données complet est passé dans le modèle pendant l'entrainement
#verbose = les logs, 0 il y a rien,  1 barre avec chiffres etc.
#validation_split=à 0,2 (20%), alors 20% des données d'entraînement seront utilisées pour la validation pendant l'entraînement du modèle, et les 80% restants seront utilisés pour l'entraînement.
# Cela permet de vérifier la performance du modèle sur des données qu'il n'a pas vues pendant l'entraînemen
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30, verbose=1, validation_split=0.2)

model.save("captcha_model.h5")
