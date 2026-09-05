# Plan d'implémentation — ALS et Stagewise ALS

## 1. Objectif

Remplacer `grow_network_svd` par une fonction qui estime une correction de rang faible

\[
X=\Omega A^\top\in\mathbb{R}^{q\times d},
\]

à partir des résidus de Bellman du batch, puis utilise cette correction pour initialiser les nouveaux neurones du réseau.

- \(q\) : nombre d'actions ;
- \(d\) : dimension des caractéristiques augmentées du biais ;
- \(d_a\) : rang maximal demandé, donc nombre maximal de neurones à ajouter ;
- \(N\) : taille du batch.

La fonction proposera deux méthodes : `"als"` et `"stagewise"`. La méthode par défaut sera `"als"`.

## 2. Interface publique

La signature existante est conservée pour limiter les changements dans le reste du projet. Les nouveaux paramètres seront optionnels et placés après `*`.

```python
def grow_network_als(
    old_net: Q_Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    d_a: int,
    numerical_threshold: float = 1e-6,
    *,
    method: str = "als",
    relative_accuracy: float,
    min_relative_loss_improvement: float = 1e-6,
    ridge: float = 0.0,
    line_search_initial_step: float = 1.0,
    line_search_armijo_alpha: float = 0.1,
    line_search_reduction: float = 0.5,
    line_search_max_iter: int = 20,
) -> tuple[Q_Network, np.ndarray]:
```

Le nom `grow_network_als` remplacera simplement `grow_network_svd` au point d'appel.

## 3. Préparation des données

La fonction principale effectue une seule fois les opérations communes aux deux méthodes.

1. Placer `states` sur le même appareil et dans le même type que le réseau.
2. Calculer sans gradient les caractéristiques fixes \(b_{l-2}(s_i)\).
3. Ajouter la coordonnée constante du biais :

   \[
   \bar b_i=
   \begin{bmatrix}
   b_{l-2}(s_i)\\
   1
   \end{bmatrix}
   \in\mathbb{R}^{d}.
   \]

4. Dans le rapport, ces vecteurs sont empilés en colonnes dans \(\bar B\in\mathbb{R}^{d\times N}\). Dans le code, le tenseur `features` aura la forme `(N, d)` : il représente donc \(\bar B^\top\), avec une observation par ligne.
5. Calculer les sorties de l'ancien réseau et les résidus associés aux actions observées :

   \[
   \widetilde r_i=	exttt{td\_targets}_i-Q_{\mathrm{old}}(s_i,a_i).
   \]

6. Vérifier les dimensions, les indices d'action et les paramètres numériques.

## 4. Opérateur d'observation

On définit

\[
H_i=e_{a_i}\bar b_i^\top\in\mathbb{R}^{q\times d},
\qquad
\mathcal H(X)_i=\frac{1}{\sqrt N}\langle H_i,X\rangle_F
=\frac{1}{\sqrt N}X_{a_i,:}\bar b_i.
\]

Deux fonctions dédiées seront écrites, car \(\mathcal H\) et \(\mathcal H^*\) sont utilisés par l'initialisation spectrale et Stagewise ALS.

### `_apply_H`

```python
def _apply_H(X, features, actions):
    # X:        (q, d)
    # features: (N, d), représentation de B_bar^T
    # actions:  (N,)
    # sortie:   (N,)
    return (X[actions] * features).sum(dim=1) / sqrt(N)
```

### `_apply_H_adjoint`

Pour \(z\in\mathbb{R}^N\),

\[
\mathcal H^*(z)
=\frac{1}{\sqrt N}\sum_{i=1}^N z_i e_{a_i}\bar b_i^\top.
\]

```python
def _apply_H_adjoint(z, features, actions, n_actions):
    # Accumuler z[i] * features[i] dans la ligne actions[i].
    # sortie: (q, d)
```

L'implémentation utilisera `index_add_`, sans construire les matrices \(H_i\).

## 5. Résolution des deux sous-problèmes ALS

Le facteur \(1/N\) est conservé dans la définition de l'objectif, mais n'a pas
d'effet sur son minimiseur. Il peut donc être omis lors de la résolution des
moindres carrés.

### 5.1 Mise à jour de \(A\) à \(\Omega\) fixé

On pose \(W=A^\top\in\mathbb{R}^{K\times d}\), \(w=\operatorname{vec}(W)\), et

\[
\omega_i=\Omega_{a_i,:}^\top\in\mathbb{R}^{K}.
\]

On construit \(D_\Omega\in\mathbb{R}^{N\times Kd}\) dont la ligne \(i\) est

\[
(D_\Omega)_{i,:}
=\operatorname{vec}(\omega_i\bar b_i^\top)^\top.
\]

Cette définition donne

\[
(D_\Omega)_{i,:}w
=\omega_i^\top W\bar b_i
=\Omega_{a_i,:}A^\top\bar b_i.
\]

Le sous-problème devient

\[
\min_w\frac1N\|\widetilde r-D_\Omega w\|_2^2.
\]

- Solution théorique si \(D_\Omega^\top D_\Omega\) est inversible :

  \[
  w=(D_\Omega^\top D_\Omega)^{-1}D_\Omega^\top\widetilde r.
  \]

- Solution générale de norme minimale :

  \[
  w=D_\Omega^\dagger\widetilde r.
  \]

- En pratique sans régularisation : pseudo-inverse calculée par SVD compacte,
  afin de gérer les matrices déficientes en rang aussi bien sur CPU que sur
  CUDA. Sur CUDA, le seul driver de `torch.linalg.lstsq` suppose en effet le
  plein rang.

On reconstruit ensuite \(W\) avec la même convention de vectorisation, puis \(A=W^\top\).

Cette opération sera isolée dans `_update_A`.

### 5.2 Mise à jour de \(\Omega\) à \(A\) fixé

Pour chaque observation,

\[
g_i=A^\top\bar b_i\in\mathbb{R}^{K}.
\]

Pour une action \(a\), on définit

\[
I_a=\{i:a_i=a\},
\]

ainsi que \(G_a\in\mathbb{R}^{|I_a|\times K}\), dont la ligne associée à \(i\in I_a\) est \(g_i^\top\). Le vecteur \(\widetilde r_a\) contient les résidus de ces mêmes observations.

La séparation par action est exacte, car l'observation \(i\) ne dépend que de la ligne \(\Omega_{a_i,:}\) :

\[
\min_\Omega\frac1N\sum_{a=1}^{q}
\|\widetilde r_a-G_a\Omega_{a,:}^\top\|_2^2.
\]

Pour chaque action présente dans le batch :

- solution théorique en plein rang :

  \[
  \Omega_{a,:}^\top=(G_a^\top G_a)^{-1}G_a^\top\widetilde r_a ;
  \]

- solution générale de norme minimale :

  \[
  \Omega_{a,:}^\top=G_a^\dagger\widetilde r_a ;
  \]

- en pratique sans régularisation : même résolution par SVD compacte que pour
  la mise à jour de (A).

Si une action est absente du batch, sa ligne est fixée à zéro, conformément à la convention de norme minimale. Le paramètre `ridge` sera prévu mais fixé à `0.0` pour le moment.

Cette opération sera isolée dans `_update_Omega`.

## 6. ALS

La fonction `_run_als` recevra les données, le rang visé, le nombre maximal d'itérations et éventuellement une initialisation de \(\Omega\).

### Initialisation spectrale

En l'absence d'initialisation fournie, on calcule

\[
Z=\mathcal H^*\!\left(\frac{\widetilde r}{\sqrt N}\right)
=\frac1N\sum_{i=1}^N\widetilde r_i e_{a_i}\bar b_i^\top.
\]

Dans le modèle exact \(\widetilde r=\sqrt N\,\mathcal H(X_\star)\), cette quantité vaut

\[
Z=\mathcal H^*\mathcal H(X_\star).
\]

On calcule les principaux vecteurs singuliers gauches de \(Z\) et on initialise \(\Omega^{(0)}\) avec les \(K\) premiers, comme dans AltMinSense.

### Itérations

À chaque itération :

1. `A = _update_A(Omega, ...)` ;
2. `Omega = _update_Omega(A, ...)` ;
3. calculer

   \[
   X=\Omega A^\top,
   \qquad
   L(X)=\frac1N\sum_{i=1}^N
   \left(\widetilde r_i-X_{a_i,:}\bar b_i\right)^2 ;
   \]

4. arrêter si l'amélioration relative de la perte est inférieure à
   `min_relative_loss_improvement`.

Le budget d'itérations est déterminé à partir de la précision relative demandée. Pour le critère retenu,

\[
T>2\log\!\left(\frac1{\texttt{relative\_accuracy}}\right),
\]

donc

```python
max_iter = floor(2 * log(1 / relative_accuracy)) + 1
```

afin de satisfaire l'inégalité stricte.

## 7. Backtracking line search

Le backtracking sera écrit dans une fonction `_backtracking_line_search` distincte.

Pour une matrice courante \(X\), un gradient \(G\) et un rang \(k\), elle teste

\[
X_{\mathrm{candidate}}=\mathcal P_k(X-\eta G),
\]

où \(\mathcal P_k\) est la projection de rang \(k\), obtenue par SVD tronquée.

1. Initialiser \(\eta=1\).
2. Construire le candidat projeté
   \(Y(\eta)=\mathcal P_k(X-\eta\nabla\psi(X))\).
3. Poser \(\Delta X(\eta)=Y(\eta)-X\), puis accepter si la condition
   d'Armijo projetée est satisfaite :

   \[
   \psi(Y(\eta))
   \leq
   \psi(X)+\alpha\langle\nabla\psi(X),\Delta X(\eta)\rangle_F,
   \qquad \alpha\in(0,0.5).
   \]

4. Sinon remplacer \(\eta\) par `0.5 * eta`.
5. Répéter au plus 20 fois.

La fonction retourne le pas accepté et le candidat projeté déjà calculé. Si
aucun pas n'est accepté après les 20 essais, elle émet un `RuntimeWarning` et
retourne `(0.0, X)` pour indiquer explicitement qu'aucune mise à jour n'a été
effectuée.

Ce choix remplace le pas fixe \(3/4\) de l'article SVP, dont la justification repose sur une condition RIP qui n'est pas établie pour l'opérateur d'observation du problème RL.

## 8. Stagewise ALS

La fonction `_run_stagewise_als` suit les étapes de l'article, avec nos notations.

1. Initialiser \(X_0=0\).
2. Pour chaque rang \(k=1,\ldots,d_a\) :

   a. Calculer le gradient de

   \[
   \psi(X)=\left\|\mathcal H(X)-\frac{\widetilde r}{\sqrt N}\right\|_2^2
   \]

   par

   \[
   \nabla\psi(X_{k-1})
   =2\mathcal H^*\!\left(
   \mathcal H(X_{k-1})-\frac{\widetilde r}{\sqrt N}
   \right).
   \]

   b. Effectuer une étape SVP de rang \(k\), avec le pas choisi par `_backtracking_line_search` :

   \[
   Y_k=\mathcal P_k\left(X_{k-1}-\eta_k\nabla\psi(X_{k-1})\right).
   \]

   c. Calculer la SVD tronquée \(Y_k=U_k\Sigma_kV_k^\top\).

   d. Initialiser le facteur gauche avec la factorisation équilibrée
   \(\Omega^{(0)}=U_k\Sigma_k^{1/2}\), puis lancer `_run_als` au rang \(k\).
   La première mise à jour de \(A\) résout le sous-problème conditionnel
   associé à cette initialisation.

   e. Poser \(X_k=\Omega A^\top\).

3. Retourner les facteurs obtenus au dernier rang effectivement traité.

Le nombre d'itérations ALS de chaque étape utilise le même budget logarithmique pratique. Il ne sera pas présenté comme une constante théorique exacte pour Stagewise ALS, car l'article ne donne qu'un ordre logarithmique avec constante cachée.

## 9. SVD finale et nombre de neurones ajoutés

Après ALS ou Stagewise ALS, on forme directement

\[
X=\Omega A^\top\in\mathbb{R}^{q\times d}.
\]

Aucune décomposition QR n'est utilisée. Comme \(q\) est le nombre d'actions, cette matrice est normalement petite.

On calcule une SVD complète économique :

\[
X=U\Sigma V^\top.
\]

Le rang numérique est

\[
r_X=\#\{j:\sigma_j>\texttt{numerical\_threshold}\},
\]

et le nombre de neurones réellement ajoutés est

\[
d_{\mathrm{added}}=\min(d_a,r_X).
\]

On conserve les \(d_{\mathrm{added}}\) premières composantes et on utilise la factorisation équilibrée

\[
\Omega_{\mathrm{final}}
=U_{:,1:d_{\mathrm{added}}}\Sigma_{1:d_{\mathrm{added}}}^{1/2},
\]

\[
A_{\mathrm{final}}
=V_{:,1:d_{\mathrm{added}}}\Sigma_{1:d_{\mathrm{added}}}^{1/2}.
\]

Ainsi,

\[
\Omega_{\mathrm{final}}A_{\mathrm{final}}^\top
=\mathcal P_{d_{\mathrm{added}}}(X).
\]

Cette unique SVD sert donc à la fois à déterminer le rang numérique et à produire les facteurs injectés dans le réseau.

## 10. Construction du nouveau réseau

La construction reste dans la fonction principale.

1. Créer un réseau avec `old_h + d_added` neurones dans l'encodeur.
2. Copier les paramètres de l'extracteur fixe `h1`.
3. Copier les anciens poids et biais de l'encodeur.
4. Copier les anciens poids et biais de la tête Q.
5. Décomposer la dernière coordonnée de \(A_{\mathrm{final}}\), qui correspond au biais :

   ```python
   W_added = A_final[:-1, :].T
   b_added = A_final[-1, :]
   ```

6. Initialiser les nouveaux neurones de l'encodeur avec `W_added` et `b_added`.
7. Initialiser leurs connexions vers la tête Q avec `Omega_final`.
8. Retourner le nouveau réseau et les valeurs singulières retenues.

## 11. Découpage retenu

Le code sera organisé autour des fonctions suivantes :

```text
grow_network_als
├── _apply_H
├── _apply_H_adjoint
├── _update_A
├── _update_Omega
├── _run_als
├── _backtracking_line_search
└── _run_stagewise_als
```

La préparation des données, la SVD finale et la construction du réseau restent dans `grow_network_als`. Ce découpage évite une fonction monolithique sans multiplier les petites fonctions artificielles.

## 12. Points arrêtés avant l'implémentation

- `method="als"` par défaut ;
- résolution par SVD compacte lorsque `ridge=0`; pour `ridge>0`, résolution du
  système augmenté de plein rang avec `torch.linalg.lstsq` ;
- `min_relative_loss_improvement=1e-6` ;
- `ridge=0.0` pour le moment ;
- budget d'itérations déduit de `relative_accuracy` par le critère logarithmique ;
- backtracking d'Armijo projeté séparé, avec `alpha=0.1`, pas initial
  `1.0`, réduction `0.5` et au plus 20 essais ;
- nombre de neurones ajouté : \(\min(d_a,\operatorname{rang}_{\mathrm{num}}(X))\) ;
- formation explicite de \(X=\Omega A^\top\) et SVD finale directe ;
- aucun test automatisé dans cette première étape d'implémentation.

Le seul choix d'interface restant à fixer avant le code est la valeur par défaut de `relative_accuracy`.
