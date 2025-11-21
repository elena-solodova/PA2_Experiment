# PA2_Experiment

Dieses Repository enthält Code und Modelle, die zum Experiment aus der PA2 gehören. Hier wird ein SVM-Modell lokal anhand der segmentierten Gehirnvolumina trainiert. Zusätzlich ist ein Skript enthalten, das für das Training des SVMs mit Swarm Learning verwendet wurde. Beide Modelle werden evaluiert.

- `models/`
  - Enthält vortrainierte Modelle (`AD_model_swarm.pth`, `linear_svm_brain_volumes.pt`).
- `src/`
  - `swarm_learning.py`: Skript für das Training in Swarm Learning.
  - `local_model_training.ipynb`: Jupyter Notebook für lokales Training und Evaluierung.
