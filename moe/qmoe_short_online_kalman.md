# Apprentissage online du scénario `short`

## 1. Modèle actuel

Pour un QMoE `m` contenant `E = 256` experts, le scénario `short` prédit la
popularité du token `t` avec :

```text
P[t,m] = W0[m] @ P[t-1,m] + W1[m] @ S[t-1,m]
```

avec :

- `S[t,m]` : activation observée des experts, de dimension `E` ;
- `P[t,m]` : popularité prédite, de dimension `E` ;
- `W0[m]` et `W1[m]` : matrices `E x E` ;
- tous les coefficients de `W0` et `W1` sont positifs ;
- la norme L1 maximale des lignes de `W0` est limitée à `0.999`.

En concaténant les deux entrées :

\[
x_t =
\begin{bmatrix}
P_{t-1,m} \\
S_{t-1,m}
\end{bmatrix}
\in \mathbb{R}^{2E},
\qquad
W_m =
\begin{bmatrix}
W0_m & W1_m
\end{bmatrix}
\in \mathbb{R}^{E\times 2E},
\]

le modèle devient :

\[
\widehat S_{t,m}=P_{t,m}=W_m x_t.
\]

L'apprentissage batch actuel minimise notamment :

\[
\mathcal L_{\mathrm{prediction}}
=
\frac{1}{TM}
\sum_{t,m}
\left\lVert P_{t,m}-S_{t,m}\right\rVert_2^2.
\]

## 2. Mise à jour online de type Kalman/RLS

À chaque nouveau token, on effectue les étapes suivantes :

1. construire \(x_t\) avec la prédiction et l'activation du token précédent ;
2. calculer \(P_{t,m}=W_{t-1,m}x_t\) ;
3. observer l'activation réelle \(S_{t,m}\) ;
4. corriger immédiatement \(W_m\).

Pour une covariance complète \(C_{t,m}\in\mathbb R^{2E\times2E}\), le gain de
Kalman/RLS est :

\[
K_{t,m}
=
\frac{C_{t-1,m}x_t}
     {R+x_t^\top C_{t-1,m}x_t},
\]

où \(R>0\) représente le bruit d'observation ou la régularisation.

L'erreur d'innovation est :

\[
e_{t,m}=S_{t,m}-W_{t-1,m}x_t.
\]

La matrice de prédiction est corrigée par :

\[
W_{t,m}
=
W_{t-1,m}
+
e_{t,m}K_{t,m}^\top.
\]

La covariance est ensuite mise à jour avec un facteur d'oubli
\(\gamma\in(0,1]\) :

\[
C_{t,m}
=
\frac{1}{\gamma}
\left(
C_{t-1,m}
-
K_{t,m}x_t^\top C_{t-1,m}
\right).
\]

- \(\gamma=1\) conserve toute l'histoire avec le même poids ;
- \(\gamma<1\) permet au modèle de s'adapter à une évolution récente du
  routage ;
- plus \(R\) est grand, plus les corrections sont prudentes.

## 3. Version diagonale recommandée

Une covariance complète demande une matrice `512 x 512` par QMoE. Une
approximation diagonale est moins coûteuse et permet de conserver une
incertitude différente pour chaque coefficient.

Pour la ligne \(i\) de \(W_m\), on note :

\[
\theta_{t,m,i}=W_{t,m}[i,:]\in\mathbb R^{2E},
\qquad
c_{t,m,i}\in\mathbb R^{2E}.
\]

La variance prédictive scalaire vaut :

\[
d_{t,m,i}
=
R_{t,m,i}
+
\sum_j c_{t-1,m,i,j}x_{t,j}^2.
\]

Le gain diagonal est :

\[
K_{t,m,i,j}
=
\frac{c_{t-1,m,i,j}x_{t,j}}
     {d_{t,m,i}}.
\]

Avec l'innovation :

\[
e_{t,m,i}
=
y_{t,m,i}
-
\sum_j\theta_{t-1,m,i,j}x_{t,j},
\]

les coefficients sont corrigés par :

\[
\theta_{t,m,i,j}
=
\theta_{t-1,m,i,j}
+
K_{t,m,i,j}e_{t,m,i}.
\]

La covariance diagonale devient :

\[
c_{t,m,i,j}
=
\frac{1}{\gamma}
\left(
c_{t-1,m,i,j}
-
\frac{c_{t-1,m,i,j}^2x_{t,j}^2}
     {d_{t,m,i}}
\right).
\]

Cette approximation coûte \(O(ME(2E))\) par token et stocke autant de
variances que de coefficients.

## 4. Intégration de la rétention

Dans `short`, la rétention utilise les 64 tokens précédents :

\[
h_{t,m,i}
=
\sum_{k=1}^{64}
\frac{\lambda}{k+\lambda}
S_{t-k,m,i}.
\]

Pour un expert donné, la composante de la perte est :

\[
\mathcal L_{t,m,i}
=
(P_{t,m,i}-S_{t,m,i})^2
+
h_{t,m,i}(P_{t,m,i}-1)^2.
\]

Cette expression est équivalente, à une constante indépendante de \(P\) près,
à :

\[
\mathcal L_{t,m,i}
=
(1+h_{t,m,i})
\left(P_{t,m,i}-y^{\mathrm{eff}}_{t,m,i}\right)^2
+
\mathrm{constante},
\]

avec la cible effective :

\[
y^{\mathrm{eff}}_{t,m,i}
=
\frac{S_{t,m,i}+h_{t,m,i}}
     {1+h_{t,m,i}}.
\]

La rétention peut donc être représentée dans le filtre par :

\[
y_{t,m,i}=y^{\mathrm{eff}}_{t,m,i},
\qquad
R_{t,m,i}
=
\frac{R}{1+h_{t,m,i}}.
\]

Un expert récemment activé produit un \(h_{t,m,i}\) élevé : sa cible effective
se rapproche de 1 et son bruit d'observation diminue, donc la correction du
filtre est plus forte.

## 5. Projection après chaque mise à jour

Une mise à jour Kalman/RLS ne garantit pas les contraintes de `short`. Après
chaque correction, il faut appliquer :

\[
W0_m\leftarrow\max(W0_m,0),
\qquad
W1_m\leftarrow\max(W1_m,0).
\]

Pour chaque ligne \(i\) de \(W0_m\) :

\[
W0_m[i,:]
\leftarrow
W0_m[i,:]\,
\min\left(
1,
\frac{0.999}
     {\sum_j W0_m[i,j]}
\right).
\]

Cette projection maintient des coefficients positifs et empêche la récurrence
de \(P_t\) de devenir instable.

## 6. Initialisation possible

Une initialisation cohérente avec le scénario actuel est :

\[
W0_{0,m}=\alpha I,
\qquad
W1_{0,m}=(1-\alpha)I,
\]

avec \(\alpha=0.719428897297\).

Les covariances diagonales peuvent être initialisées avec :

\[
c_{0,m,i,j}=\sigma_0^2,
\]

où une grande valeur de \(\sigma_0^2\) autorise un apprentissage initial
rapide. Les paramètres principaux à étudier sont alors :

- le facteur d'oubli \(\gamma\) ;
- le bruit de base \(R\) ;
- la variance initiale \(\sigma_0^2\) ;
- une éventuelle borne maximale du gain \(K\).

## 7. Limite du scénario `future`

Cette transformation est strictement online pour `short`, car toutes ses
entrées et sa rétention sont disponibles au token courant. Le scénario
`future`, qui utilise également les activations des 16 tokens suivants, ne
peut être appris de manière strictement online sans retarder les mises à jour
d'au moins 16 tokens.
