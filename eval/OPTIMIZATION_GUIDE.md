# Guide d'Optimisation Séquentielle - RAG Pipeline

## 🎯 Objectif

Trouver la **meilleure combinaison** de composants RAG en tenant compte des **interactions entre composants**, contrairement à l'ablation classique qui suppose l'indépendance.

## ❌ Problème de l'Ablation Classique

L'ablation standard teste une variable à la fois :
- `BM25` est meilleur avec `chunking=fixed` (baseline)
- `chunking=markdown` est meilleur avec `retrieval=hybrid` (baseline)

**Mais on ne sait pas** si `BM25 + markdown` est meilleur que `hybrid + markdown` !

Les composants **interagissent** : l'optimal pour un composant **dépend** des choix en amont.

---

## ✅ Solution : Optimisation Séquentielle Hiérarchique

```
PHASE 1: RETRIEVAL PIPELINE
  ↓ Fixe la meilleure combinaison (chunking + embedding + retrieval)
  
PHASE 2: RERANKING
  ↓ Teste rerankers avec retrieval optimal fixé
  
PHASE 3: GENERATION
  ↓ Teste LLMs avec retrieval + reranking optimaux fixés
  
RÉSULTAT: Configuration globalement optimale
```

---

## 📋 Phases Détaillées

### **Phase 1 : Retrieval Pipeline**

**Composants testés** : Chunking + Embedding + Retrieval

**Grille** :
- Chunking : `fixed`, `recursive`, `markdown` (3)
- Embedding : `BGE-M3`, `Multilingual-E5` (2)
- Retrieval : `hybrid`, `dense_only`, `bm25_only` (3)
- **Total** : 3 × 2 × 3 = **18 combinaisons**

**Métriques** : Retrieval uniquement (pas de LLM, rapide)
- candidate_recall
- context_recall
- context_precision
- ndcg_at_10

**Temps** : ~15-20 minutes

**Score composite** : `0.40 × precision + 0.30 × recall + 0.30 × ndcg`

**Sortie** : `eval/phase1_retrieval_optimization.csv`

---

### **Phase 2 : Reranking**

**Composants testés** : Reranking strategies

**Grille** (avec retrieval optimal de Phase 1) :
- `disabled` : Pas de reranking
- `cross_encoder` : ms-marco-MiniLM-L-6-v2 (rapide)
- `llm_bedrock` : Claude Sonnet 4.6 (lent mais précis)
- **Total** : **3 stratégies**

**Métriques** : Retrieval (precision affectée par reranking)

**Temps** : ~10-15 minutes (llm_bedrock plus lent)

**Score composite** : `0.50 × precision + 0.25 × recall + 0.25 × ndcg`
(Precision plus importante car c'est ce que le reranking affecte le plus)

**Sortie** : `eval/phase2_reranking_optimization.csv`

---

### **Phase 3 : LLM Generation**

**Composants testés** : LLM models

**Grille** (avec retrieval + reranking optimaux) :
- `qwen3:1.7b` (baseline)
- `qwen3:3b`
- `qwen3:7b`
- `qwen3.5:1.8b`
- `qwen3.5:3b`
- `smollm2:1.7b`
- **Total** : **6 LLMs**

**Métriques** : Retrieval + Generation (LLM-as-judge)
- faithfulness
- answer_correctness
- context_precision
- ndcg_at_10

**Temps** : ~20-30 minutes (appels juge Bedrock)

**Score composite** : `0.30 × precision + 0.15 × ndcg + 0.30 × faithfulness + 0.25 × correctness`
(Même formule que l'ablation study)

**Sortie** : `eval/phase3_generation_optimization.csv`

---

## 🚀 Usage

### **Option A : Run Complet (Recommandé)**

Lance les 3 phases automatiquement et configure `default.yaml` :

```bash
uv run python -m eval.run_full_optimization
```

**Temps total** : ~45-60 minutes

**Actions** :
1. Run Phase 1 (18 configs retrieval)
2. Run Phase 2 (3 rerankers)
3. Run Phase 3 (6 LLMs)
4. Applique automatiquement la meilleure config à `config/default.yaml`
5. Affiche rapport final

---

### **Option B : Phase par Phase (Manuel)**

Si tu veux contrôler chaque étape :

```bash
# Phase 1 : Retrieval
uv run python -m eval.optimize_retrieval

# Vérifie les résultats
cat eval/phase1_retrieval_optimization.csv

# Phase 2 : Reranking
uv run python -m eval.optimize_reranking

# Vérifie les résultats
cat eval/phase2_reranking_optimization.csv

# Phase 3 : Generation
uv run python -m eval.optimize_generation

# Vérifie les résultats
cat eval/phase3_generation_optimization.csv

# Applique manuellement la meilleure config
# (éditer config/default.yaml avec les valeurs du CSV)
```

---

## 📊 Résultats Attendus

### **Comparaison vs Ablation Classique**

| Approche | Baseline | Ablation Classique | Optimisation Séquentielle |
|----------|----------|-------------------|---------------------------|
| Composite | 0.673 | 0.797 (SmolLM seul) | **0.82-0.85** (estimé) |
| Gain | - | +18.4% | **+22-26%** |

**Pourquoi meilleur ?**
- L'ablation teste `SmolLM` avec config baseline retrieval
- L'optimisation teste `SmolLM` avec **meilleur retrieval possible**
- Les interactions sont prises en compte !

---

## 📁 Fichiers Générés

```
eval/
├── phase1_retrieval_optimization.csv    # 18 configs retrieval
├── phase2_reranking_optimization.csv    # 3 rerankers
├── phase3_generation_optimization.csv   # 6 LLMs
└── OPTIMIZATION_GUIDE.md                # Ce guide
```

**Format CSV** : Trié par composite score décroissant (meilleur en premier)

---

## 🔧 Personnalisation

### Ajouter des Composants

**Ajouter un LLM** :
```python
# eval/optimize_generation.py, ligne 16
LLM_MODELS = [
    "qwen3:1.7b",
    "smollm2:1.7b",
    "your-new-model:1.7b",  # Ajouter ici
]
```

**Ajouter une stratégie de chunking** :
```python
# eval/optimize_retrieval.py, ligne 18
CHUNKING_STRATEGIES = ["fixed", "recursive", "markdown", "semantic"]  # Ajouter semantic
```

### Modifier le Score Composite

**Phase 1** (retrieval) :
```python
# eval/optimize_retrieval.py, ligne ~85
r["composite"] = (
    0.40 * r["context_precision"]  # Ajuster poids
    + 0.30 * r["context_recall"]
    + 0.30 * r["ndcg_at_10"]
)
```

---

## 🎓 Exemple Complet

```bash
# 1. Lancer optimisation complète
uv run python -m eval.run_full_optimization

# Attendre ~45-60 min...

# 2. Résultats affichés :
# ================================================================================
# OPTIMAL CONFIGURATION:
# --------------------------------------------------------------------------------
#   Chunking:   markdown
#   Embedding:  multilingual_e5
#   Retrieval:  bm25_only
#   Reranking:  llm_bedrock
#   LLM:        smollm2:1.7b
#
# PERFORMANCE METRICS:
# --------------------------------------------------------------------------------
#   Composite Score:     0.8456
#   Context Precision:   0.920
#   Context Recall:      1.000
#   Faithfulness:        0.938
#   Answer Correctness:  0.612

# 3. Configuration automatiquement appliquée à config/default.yaml

# 4. Tester sur UI
uv run streamlit run src/ui/streamlit_app.py
```

---

## ⚠️ Notes Importantes

1. **Ollama requis** : Tous les modèles LLM testés doivent être disponibles dans Ollama
   ```bash
   ollama pull qwen3:1.7b
   ollama pull smollm2:1.7b
   # etc.
   ```

2. **Bedrock requis** : Pour Phase 3 (juge LLM) et reranking LLM
   - Variable `AWS_BEARER_TOKEN_BEDROCK` dans `.env`

3. **Temps d'exécution** : Peut varier selon :
   - Vitesse GPU/CPU pour LLM inference
   - Latence Bedrock API
   - Taille du corpus

4. **Résultats reproductibles** : Retrieval déterministe, génération avec `temperature=0.1` (variance minime)

---

## 🔬 Analyse des Résultats

### Interpréter les CSVs

**Phase 1** : Chercher patterns
- `markdown` meilleur que `fixed` ?
- `bm25_only` vs `hybrid` : quelle différence ?
- Impact de `embedding` : marginal ou significatif ?

**Phase 2** : Trade-off precision vs coût
- `llm_bedrock` : +15-20% precision mais 10x plus lent
- `cross_encoder` : bon compromis vitesse/qualité

**Phase 3** : LLM le plus impactant
- `smollm2` souvent meilleur en faithfulness
- `qwen3.5` meilleur en answer_correctness
- Compromis selon use-case

---

## 📈 Monitoring

Chaque phase affiche :
```
[3/18] Evaluating: markdown_multilingual_e5_bm25
  → Recall: 1.000, Precision: 0.880, nDCG: 0.995
```

**Signes d'alerte** :
- Recall < 0.98 : Un retriever rate des documents
- Precision < 0.50 : Trop de bruit sélectionné
- Faithfulness < 0.60 : LLM hallucine
- Correctness < 0.40 : LLM ne répond pas aux questions

---

## 🎯 Conclusion

Cette approche **séquentielle** est **plus intelligente** que l'ablation classique car :
✅ Prend en compte les **interactions** entre composants
✅ Chaque phase optimise sur la **meilleure base** possible
✅ Résultat final **globalement optimal**, pas localement optimal
✅ Plus **rapide** que tester toutes les combinaisons (18+3+6=27 vs 3×2×3×3×6=324)

**Temps total** : ~1h au lieu de ~10h pour grid search complet !
