# Few Shot Concept Learning in LMs

This repo is organized into the following subdirectories:

- `src` contains scripst to run all experiments, along with helper functions
    - `eval_model.py` runs behavioral analysis on the concept list task. This supports Fig. 2
    - `harvest_model_data.py` extracts representations and logits from models on the concept list task.
    - `latent_regressions.py` attempts to learn an accentuation matrix, rotation matrix, or MLP to predict context displacement vectors. This supports Fig. 4 and Appendix figures.
    - `behavioral_modeling.py` attempts to fit an exponential tilt model using an accentuation matrix or raw prototypes. This supports Fig. 5.
    - `context_displacement_analysis.py` analyzes aspects of the context displacement vector. This supports Fig. 6 and Appendix figures.
    - `feature_steering.py` performs causal steering interventions. This supports Fig. 7.
    - `concept_stretching.py` performs an analysis of the singular vectors of the accentuation matrix. This supports Table 1.
- `analysis` contains all ipynbs, used to generate figures and tables.
- `data` contains the concepts data.
- `results` contains our results files.
