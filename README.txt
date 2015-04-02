Modifications du code FaceDetect - Camshift(Garance Vallat)

Testé avec une webcam dont l'image est de taille 640 * 480.
Testé avec openCV version 2.4.9
Testé avec Ubuntu 14.04


Version 1
Ce qui a été retiré
Le cas où une image ou une vidéo est passée en paramètre lors de l'exécution du programme a été retiré : cette possibilité n'existe donc plus. 
Le booléen tryflip et toutes ses références a été retiré, étant donné que ma webcam ne gère pas de mode miroir. 
Toutes les références à nestedObjects ont été retirées, pour se concentrer seulement sur la détection d'un visage, et pas sur les traits plus précis. 

Ce qui a été ajouté
Deux paramètres on été ajoutés à la méthode detectandDraw : 
Il s'agit de deux rectangles, roi et une référence vers previousRoi. 
ROI est la zone d'intérêt dans laquelle l'algorithme de reconnaissance doit effectuer sa recherche pour le visage. 
PreviousRoi est le dernier rectangle à l'intérieur duquel l'algorithme a reconnu un visage, si la dernière itération a permis d'en reconnaître un. 
Une méthode checkBoundaries a également été ajoutée : elle concerne les calculs des ROI : on calcule les coordonnées désirées pour notre nouveau rectangle, mais avant de le créer, on vérifie qu'il ne dépasse pas des bords de l'image, auquel cas on le limite au bord de l'image sur le côté concerné. 

Lors de l'appel à la méthode detectAndDraw, on distingue plusieurs cas : 
Lors de la toute première itération, la zone d'intérêt est l'image entière. 
Ensuite, on va rencontrer le cas où un visage a été reconnu dans l'image à l'itération précédente : le rectangle previousRoi contient donc des coordonnées, qui sont à priori les plus précises pour pouvoir retrouver encore un visage : il devient donc notre nouvelle zone d'intérêt. 

Lorsqu'aucun visage n'est détecté, on définit une nouvelle zone d'intérêt à partir de la précédente, en l'agrandissant légèrement, pour essayer de suivre le visage qui a "disparu" pour la webcam. 

Cette méthode permet de suivre un visage qui se déplace à la caméra à une vitesse normale, tout en permettant de sauter sur le prochain visage présent si le premier disparaît complètement. 

Version 2
Ce qui a été retiré
Toutes les références à la notion d'échelle ont été retirées, aussi bien dans l'analyse des arguments passés en paramètre à l'exécution que dans l'exécution du code de faceDetect. 
On ne peut plus arrêter l'exécution en appuyant sur n'importe quelle touche : en effet, certaines sont bloquées pour permettre à l'enregistrement du fichier texte de n'avoir lieu que dans le cas où l'utilisateur le demande avec la touche espace. 

Ce qui a été ajouté
Le code principal de camshift, moins les possibilités de jouer sur les paramètres de couleur. 

La fonction de camshift est appelée avec en paramètres l'image de la webcam, le zone où facedetect a repéré le visage, et en référence l'histogramme de cette zone, qui peut ne pas avoir encore été initialisé. 
Lors de l'appel à camshift, il y a deux possibilités : 
Si c'est le premier appel, on commence par calculer l'histogramme de la zone d'intérêt. 
Ensuite, dans tous les cas : 
on décale la sélection vers la gauche, pour correspondre à une personne droitière qui lèverai la main. 
Cela nous donne un rectangle vertical qui contient la main, si la même couleur de peau a été détectée, grâce à la comparaison d'histogramme. 

Quand l'utilisateur veut enregistrer la sélection trouvée de la main, il appuie sur la barre d'espace. 
On découpe ce rectangle pour le transformer en carré : cela permet de créer une nouvelle image (matrice) qui est carrée. 
Afin d'éviter tout risque de parasitage par le visage qui serait trop proche de la main, on passe à 0 le tiers droit de l'image, dans lequel la main ne devrait pas se trouver. 
C'est cette dernière matrice qu'on enregistre dans le fichier texte. Le fichier texte ci joint décrit la lettre C. 