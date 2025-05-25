
# Iris Tracker avec OpenCV et Mediapipe

## Introduction

Cet Iris Tracker utilise OpenCV et Mediapipe pour suivre les mouvements de l'iris en temps réel. Il peut être utilisé dans plusieurs applications médicales importantes, comme la détection de troubles neurologiques, l'évaluation des fonctions cognitives, et l'amélioration de la communication pour les patients non verbaux.

## Importance dans le domaine médical

1. **Détection de troubles neurologiques**  
   Les mouvements des yeux peuvent révéler des informations sur l'état du système nerveux central. Par exemple, les saccades et les mouvements de poursuite lisse peuvent être analysés pour détecter des troubles comme la maladie de Parkinson, la sclérose en plaques, et d'autres troubles neurologiques.

2. **Suivi de la vigilance et de la somnolence**  
   Le suivi des mouvements oculaires peut aider à évaluer la vigilance et détecter la somnolence, particulièrement utile pour les patients en soins intensifs ou sous sédation.

3. **Évaluation des fonctions cognitives**  
   Les tests de suivi oculaire peuvent aider à évaluer les fonctions cognitives et la perception visuelle des patients souffrant de lésions cérébrales ou de troubles cognitifs. Cela peut être utilisé dans la rééducation et le suivi de la progression de ces conditions.

4. **Diagnostic et suivi des troubles oculaires**  
   Les mouvements irréguliers ou anormaux des yeux peuvent être des indicateurs de troubles oculaires tels que le nystagmus, le strabisme, ou d'autres anomalies. Un suivi précis peut aider à diagnostiquer et à traiter ces conditions plus efficacement.

5. **Amélioration de la communication pour les patients non verbaux**  
   Pour les patients atteints de paralysie ou de troubles de la communication, un Iris Tracker peut être utilisé comme interface de communication assistée par ordinateur, améliorant leur qualité de vie.

6. **Recherche en psychologie et neurosciences**  
   Le suivi des mouvements oculaires est utilisé pour étudier l'attention, la perception, et la prise de décision, aidant à mieux comprendre le fonctionnement du cerveau.

---

## Fonctionnalités principales

- Détection et suivi en temps réel des iris gauche et droit  
- Visualisation des contours des iris avec personnalisation de la couleur et de l’épaisseur  
- Utilisation optimisée de Mediapipe Face Mesh pour une détection précise  
- Fonctionne avec une simple webcam, sans matériel spécialisé

---

## Prérequis

- Python 3.7+  
- OpenCV  
- Mediapipe  
- Numpy

---

## Installation

Installez les dépendances nécessaires via pip :

```bash
pip install opencv-python mediapipe numpy
````

---

## Utilisation

1. Clonez le dépôt ou copiez le script Python.
2. Lancez le script :

```bash
python IrisTracker.py
```

3. La webcam s'ouvre et détecte vos iris en temps réel.
4. Appuyez sur la touche **q** pour quitter.

---

## Personnalisation

* Changez la couleur des contours des iris dans la fonction `DrawIris` en modifiant le paramètre `color`.
* Ajustez la confiance minimale de détection dans le constructeur de `IrisTracker` via `minDetectionCon`.
* Affichez les ID des points du visage en activant le paramètre `fm_id` dans `GetFaceMesh`.

---

## Limitations et améliorations futures

* Détection limitée à un visage à la fois (maxfaces=1)
* Précision influencée par la qualité de la webcam et l’éclairage
* Amélioration du suivi en cas de mouvements rapides ou occultations
* Potentiel d’intégration avec des systèmes d’interface cerveau-machine (BCI)

---

## Contributions

Les contributions sont les bienvenues ! N’hésitez pas à proposer des améliorations, corrections ou nouvelles fonctionnalités via Pull Requests.

---

## Licence

Ce projet est sous licence MIT.

---

Merci d’utiliser cet Iris Tracker pour vos projets de recherche, médicaux ou éducatifs !


