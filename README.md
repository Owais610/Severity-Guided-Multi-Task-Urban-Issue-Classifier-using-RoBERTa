# Urban Issue Classifier — Experimental Multi-Task NLP Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> [!IMPORTANT]
> **Archived experimental project — not validated for real-world use.**
>
> This repository documents an exploratory machine-learning project built to experiment with **RoBERTa**, multi-task text classification, and several training techniques. It is **not a production system, an operational civic-reporting tool, or a validated method of determining the seriousness of a complaint**.
>
> **The dataset contains synthetic/generated content, and the authenticity, provenance, and reliability of all texts and labels have not been independently established.** The proportion of synthetic content is not reliably quantified. Results obtained with this dataset **must not be interpreted as evidence of performance on authentic, real-world complaints**.
>
> **Do not use this model to triage emergencies, assign public-service priorities, make safety or healthcare decisions, or take action affecting individuals.** Its outputs are experimental predictions, not verified risk assessments.

**Project status:** Shelved indefinitely; retained as a technical and educational reference. No deployment or further research validation is planned.

---

## Overview

This project explores **multi-task text classification** using a RoBERTa-based neural network. Given a text input, the model attempts to predict two labels from the project's dataset:

- **Issue category:** one of eight predefined urban-issue categories.
- **Dataset severity label:** Low, Medium, or High.

The goal was to learn about and experiment with shared representations, classification heads, loss weighting, prototypes, and contrastive training. **The presence of a severity output does not mean the model can reliably determine real-world urgency, danger, or public-service priority.**

![Project introduction](Images/Intro.jpg)

## Scope and intended use

**Appropriate uses:** inspecting the code, reproducing the experimental setup where dependencies and data permit, understanding multi-task learning, and studying failure modes or limitations in model and dataset design.

**Not intended for:** deployment, real-time complaint monitoring, municipal workflows, emergency triage, resource allocation, healthcare or public-safety decisions, policy decisions, or decisions affecting a person's access to services.

This repository does not claim a new state-of-the-art method, independently verified labels, demonstrated real-world generalization, or operational readiness. Model outputs should not be presented as factual assessments of a complaint's severity.

---

## Model architecture

The experimental design uses the `roberta-base` encoder from Hugging Face Transformers and task-specific components:

### Category prediction

- Projects learned representations into a category-label space.
- Uses learned category embeddings and scaled cosine similarity.

### Severity prediction

- Uses an MLP classification head, described in the original design as `768 → 768 → 3`.
- Predicts one of three *dataset-defined* severity labels; these are **not independently validated measures of actual urgency**.

### Prototype-based mechanisms

- Explores severity-related prototype representations and similarity-based weighting.
- The described weighting takes the form:

  ~~~text
  W_i = sigmoid((E_i · P_s) / τ)
  ~~~

These components are part of an experimental architecture; their inclusion should not be taken as proof of improved accuracy, interpretability, or reliability.

![Model and loss overview](Images/Loss.jpg)

---

## Training techniques explored

The project combines several established modeling ideas as experiments, **without claiming that each component is necessary or provides a reproducible improvement**.

### Dynamic class-loss reweighting

Tracks class-wise losses and uses isotonic regression to estimate loss trends that inform experimental class weights. The design includes a weighting heuristic of the form:

~~~text
α_c ∝ estimated_asymptotic_loss × (1 + β × variance)
~~~

![Isotonic regression loss fit](Images/Iso_Reg_Fit.jpg)

**Important:** the dataset is not strongly class-imbalanced overall. This work does not establish that imbalance correction was needed or that the reweighting approach outperforms simpler alternatives.

### Severity-guided category-loss scaling

Experiments with weighting category loss using dataset severity labels:

~~~text
L_category_weighted = L_category × (1 + α × severity_value)
~~~

Because the severity annotations are not independently validated, **this weighting must not be interpreted as a reliable way to emphasize genuinely urgent cases**.

### Prototype separation

Explores a margin-based separation objective for learned prototypes:

~~~text
L_proto = max(0, margin - ||P_high - P_low||)
~~~

### Supervised contrastive learning

Experiments with bringing representations from the same category closer while separating different categories.

### Label smoothing

Includes label smoothing as an experimental regularization technique. The project does not establish that it consistently improves calibration or generalization.

---

## Dataset and data-quality disclosure

The original experiments used **approximately 37,000 English-language text entries** assigned to eight categories:

| Category | Category |
|---|---|
| Environmental Health | Housing & Rent |
| Waste & Sanitation | Transit Systems |
| Healthcare Access | Road Integrity |
| Public Safety | Utility Stability |

Each entry also has one of three severity labels: **Low, Medium, High**.

### Known limitations of the data

- **Synthetic/generated text is present.** The complete mix of synthetic, copied, transformed, and authentic posts has not been conclusively established.
- **Source provenance is incomplete.** Original post identifiers, source URLs, and collection metadata are not consistently available for independent verification.
- **Label reliability is uncertain.** A fully documented, independent human-annotation and adjudication process is not available for the complete dataset.
- **Some text records exhibit quality issues**, including apparent repetition within individual entries and mixtures of firsthand complaints with news or commentary.
- **The labels are project-specific.** They must not be equated with official municipal categories, independently verified public-safety severity, or actual response urgency.

Consequently, **this dataset should not be described as a verified real-world civic-complaint benchmark or as a reliable ground-truth severity dataset**. It should not be reused for external claims about operational performance without substantial independent validation.

### Original experimental split

The project used a **70% training / 15% validation / 15% test** split, stratified by category × severity. This describes the experimental setup only: a stratified split **does not remedy uncertain labels, synthetic contamination, potential source overlap, or shortcomings in external validation**.

---

## Training workflow (experimental)

The original pipeline was organized into the following stages:

1. **Warm-up:** train the category head with an initially frozen backbone and class-balanced focal loss.
2. **Class-loss analysis:** fit isotonic regression to experimental loss trajectories and derive reweighting factors.
3. **Initialization:** estimate severity centroids and initialize prototype and category representations.
4. **Joint training:** optimize category, severity, prototype, and contrastive objectives.

The general loss described by the project is:

~~~text
L_total =
    λ_cbf      × L_CBF +
    λ_severity × L_severity +
    λ_proto    × L_prototype +
    λ_contrast × L_contrast
~~~

See the implementation for active settings and precise formulas. **The architecture and training choices are presented as experiments, not as established methodological innovations.**

---

## Evaluation and interpretation

The experimental code reports:

| Task | Metrics |
|---|---|
| Category classification | Macro-F1, weighted-F1, accuracy |
| Severity classification | Accuracy, ordinal mean absolute error (MAE) |
| Joint classification | Proportion with both dataset labels predicted correctly |

Artifacts may include training curves, confusion matrices, learned weights, and JSON summaries.

**Interpretation warning:** these metrics compare predictions against labels in the project's dataset. They do **not** establish performance on new cities, real complaints, genuine emergencies, or accurately human-verified severity levels. Plots of attention or prototypes are also **not explanations or validation of decision-making** by themselves.

No real-world performance, fairness, or safety claim should be based solely on these results.

---

## Running the experiment

This section is included to document the original development workflow, **not to recommend using the trained model for real-world predictions**.

### 1. Install dependencies

~~~bash
pip install transformers datasets torch torchmetrics scikit-learn pandas numpy matplotlib seaborn tqdm coral-pytorch
~~~

### 2. Prepare compatible input data

The original code expects a CSV structure similar to:

~~~csv
Post,Category,Severity
"Example text for a controlled experiment",Housing & Rent,High
~~~

**This is an illustrative format, not an example of an authenticated complaint or verified severity annotation.** Use appropriately licensed and ethically sourced data for any independent experiments.

### 3. Configure the input path

~~~python
DATA_PATH = "path/to/your/dataset.csv"
~~~

### 4. Run the training code

If the corresponding script exists in your checkout:

~~~bash
python train.py
~~~

Depending on the repository version, the pipeline may instead be organized as notebook cells. Exact dependencies and execution steps have not been independently revalidated for this archived project.

---

## Illustrative labels (not verified predictions)

The examples below only demonstrate the **format** of category/severity outputs. They are not confirmed model predictions and should not be interpreted as correct urgency judgments.

| Example text | Example category | Example severity label |
|---|---|---|
| "Garbage not collected" | Waste & Sanitation | Low |
| "Transit delays" | Transit Systems | Medium |
| "Cracked roof collapsing" | Housing & Rent | High |

The examples are schematic. In particular, a real structural or safety concern requires appropriate human assessment, **not classification by this model**.

---

## Limitations and lessons learned

- The dataset's synthetic content and incomplete provenance undermine claims of real-world validity.
- Severity-label reliability has not been independently demonstrated.
- Random in-dataset evaluation is not a substitute for external, source-disjoint validation.
- Severity outputs may collapse toward frequent labels, and prototype/attention weights may not meaningfully distinguish severity.
- Several modeling components were combined without conclusive evidence that each provides a necessary improvement over a simpler baseline.
- The project is not maintained or optimized for secure, robust, or responsible deployment.

**The main value of this repository is educational:** it preserves an exploration of transformer fine-tuning and multi-task learning, together with limitations that are important when designing future experiments. The project is **shelved indefinitely**; this repository is retained for transparency rather than advertised as a finished real-world solution.

## License

MIT (subject to the repository's `LICENSE` file). The software license does **not** certify the dataset's provenance or grant rights to third-party data, and it does not imply the model is suitable for deployment.
