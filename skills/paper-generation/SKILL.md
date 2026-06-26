# Skill: Paper Generation

## Paper Generation Pipeline

### 1. Gather Inputs
Read these files:
- `results/sota_comparison.md` — benchmark data
- `results/exp_{best_id}_report.md` — best results
- `configs/models/{best_model}.yaml` — architecture details
- `skills/lung-sound-research/SKILL.md` — related work

### 2. Generate LaTeX

LaTeX structure:
```latex
\documentclass[conference]{IEEEtran}
% ... preamble ...

\begin{document}

\title{{{TITLE}}}
\author{{{AUTHORS}}}
\maketitle

\begin{abstract}
{{ABSTRACT}}
\end{abstract}

\section{Introduction}
{{INTRODUCTION}}

\section{Related Work}
{{RELATED_WORK}}

\section{Proposed Method}
{{PROPOSED_METHOD}}

\section{Experimental Setup}
{{EXPERIMENTAL_SETUP}}

\section{Results and Discussion}
{{RESULTS}}

\section{Ablation Studies}
{{ABLATION}}

\section{Conclusion}
{{CONCLUSION}}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

### 3. Auto-Generate Figures

```python
# paper/generate_figures.py
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_figures(results_dir, paper_dir):
    # 1. Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=['Normal','Wheeze','Crackles','Both'],
                yticklabels=['Normal','Wheeze','Crackles','Both'])
    fig.savefig(f'{paper_dir}/figures/confusion_matrix.pdf')
    
    # 2. SOTA Comparison Bar Chart
    # ...
    
    # 3. Training Curves
    # ...
    
    # 4. Per-Class Radar Chart
    # ...
```

### 4. Reference Management
Auto-generate `references.bib` from Research Agent findings.

## Target Venues
- ICASSP 2025
- INTERSPEECH 2025
- EMBC 2025 (IEEE EMBS)
- IEEE JBHI (Journal)
