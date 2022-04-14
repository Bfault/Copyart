# Copyart: transformer votre image dans le style d'un artiste

<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/tour_eiffel.jpg?raw=true" alt="tour eiffel" width="80%">
<p>
<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/tour_eiffel_ukiyoe.jpg?raw=true" alt="tour eiffel Ukiyoe" width="40%">
<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/tour_eiffel_monet.jpg?raw=true" alt="tour eiffel Monet" width="40%">
</p>
<figcaption>
    <b>Images dans le style Ukiyoe et Monet</b>
</figcaption>

# Qu'est-ce que c'est ?

Copyart est un programme qui utilise l'architecture du CycleGAN pour permettre de transformer une image choisie dans le style d'un artiste.

Le CycleGAN a été élaboré en 2017 au [BAIR (Berkeley AI Research)](https://arxiv.org/abs/1703.10593), il vient des réseaux GAN.

> Les GAN (generative adversarial networks) sont des modèles générateurs où 2 réseaux de neurones sont mis en compétition, l'un (le générateur) essaye de générer des images aussi réalistes tandis que l'autre (le discriminateur) essaye de détecter si l'image est réel ou non.

Le programme va passer d'un domaine (image) à un autre domaine (oeuvre d'art).

# Pourquoi le CycleGAN ?

Le CycleGAN est différent du GAN classique, car il contient 2 générateurs et 2 discriminateurs.

Contrairement aux GAN classiques qui utilisent des données pairées, le CycleGAN utilise des données non pairées.

![test](https://miro.medium.com/max/1400/1*40iuLVgb0Xfny1zWybFArg.png)

Dans notre cas il est très dur, voir impossible de trouver une base de données avec des images réaliste et les mêmes images dans le style d'un artiste. Le CycleGAN voit sont utilité ici car il permet d'utiliser des données non pairées.

# Quels sont les livrables ?

Le programme est utilisable via un site internet pouvant être déployé et via une interface CLI.

Le site n'est actuellement pas déployé pour cause de performance.

# Le développement

Le projet peut être scindé en 2 parties :

# Partie 1 : l'entraînement

L'entraînement sert à améliorer les performances du modèle. Il faut l'entraîner pour chaque artiste individuellement.

## Les datasets

Les données ont été récupérées sur Kaggle:

- Une base de données d'oeuvre d'art, [best artwork of all time par ikarus777](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) qui contient plus de 50 artistes avec par exemple 877 peintures pour Vincent VanGogh.

- Une base de données d'images, [Flickr8K par adityajn105](https://www.kaggle.com/datasets/adityajn105/flickr8k) qui contient 8 091 images qui viennent du site Flickr.

---

> J'utilise la même architecture que dans le papier de recherche ([rappel](https://arxiv.org/pdf/1703.10593.pdf))
>- Une taille de Batch de 1
>- 100 epochs
>- Un learning rate de 0.0002
>- Un lambda cycle de 10 (défini plus tard)
>- Les architectures

---

## Le générateur

<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/cyclegan.png?raw=true" alt="schéma du générateur">

<figcaption align="center">
    <b>Architecture d'un générateur pour une résolution de 128 x 128</b>
</figcaption>

Il est dit dans le papier que 6 blocks résiduelles sont utilisés pour une résolution d'image de 128 x 128 sinon 9 pour une résolution de 256 x 256.

## Le discriminateur

Les discriminateurs utilisent l'architecture du patchGAN.

<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/patchgan.ppm?raw=true" alt="schéma du discriminateur">

<figcaption align="center">
    <b>Architecture du patchGAN</b>
</figcaption>

## Les fonctions de coût (loss functions)

Le cycleGAN contrairement à un GAN classique utilise plusieurs fonctions de coûts :

- L<sub>adv</sub> Le coût adversarial (adversarial loss) est le coût de base des GAN, celui que permet de comparer les images générées par le générateur avec les images réelles.

- L<sub>cyc</sub> Le coût de cohérence du cycle (cycle consistency loss) qui permet de garantir le lien entre l'image donnée et celle générée.

- L<sub>id</sub> Le coût d'identité (identity loss) qui permet qu'un générateur d'un domaine A vers un domaine B puisse générer une image identique si une image de domaine B est donnée.

Le coût total est égal à L<sub>total</sub> = L<sub>adv</sub> + λ<sub>cyc</sub>L<sub>cyc</sub> + λ<sub>id</sub>L<sub>id</sub>

λ<sub>cyc</sub> et λ<sub>id</sub> sont des hyperparamètres (lambda cycle et lambda id) qui sont modifiables. Dans notre cas lambda id est égal à 0 car cela n'est pas nécessaire (le coût d'identité sert à conserver la couleur). 

# Partie 2 : l'utilisation des modèles entraînés

Les performances demandées par l'entraînement sont trop élevées pour être réalisées par mon ordinateur, j'ai donc utilisé les modèles pré-entraînés implémentés avec le papier de recherche.

La page web a été réalisée avec Flask et ressemble à ceci :

<img src="https://github.com/Bfault/Copyart/blob/master/example/assets/site.png?raw=true" alt="site">

On peut selectionner une image avec le bouton "select from the library" et l'artiste avec le champ de sélection.

Puis transformer l'image avec le bouton "transform" et pouvoir télécharger le résultat avec le bouton "download".

# Conclusion :

Ce projet a été réalisé pour apprendre le fonctionnement du cycleGAN et a été une réussite. Cela m'a aussi permis de voir d'autres notions de deep learning tel que les padding types, les normes, les residuals blocks, etc.

Pour améliorer le projet :

- Créer un conteneur docker pour pouvoir déployer la page web (cela permet de pouvoir la lancer indépendamment de l'environnement)

- Une refonte de la page web

# Références:

- [CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf)

- [mon repo git](https://github.com/Bfault/Copyart)

- [Kaggle (pour trouver les datasets)](https://www.kaggle.com/)