# Neural Architecture Search

A lightweight **Neural Architecture Search (NAS)** project using **Optuna, TensorFlow/Keras, and Fashion-MNIST**.

The project automatically searches for a useful neural-network architecture instead of relying on a manually selected architecture. It is intentionally designed for a CPU-only computer: the search uses a small dataset subset, few trials, and short training runs.

## Features
- NAS and AutoML
- Optuna architecture optimization
- Search over hidden layers, units, dropout, learning rate, and batch size
- Fashion-MNIST classification
- Notebook-first implementation
- CPU-friendly experiment
- Django web demo
- No `src/`, preprocessing package, or separate ML module

## Structure
```text
Neural-Architecture-Search/
├── data/README.md
├── models/
├── notebooks/neural_architecture_search.ipynb
├── static/
├── templates/index.html
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── CONTRIBUTE.md
└── CHANGELOG.md
```

## Workflow
1. Fashion-MNIST is downloaded by Keras.
2. A small subset is selected.
3. Optuna creates candidate architectures.
4. Candidates are trained for a few epochs.
5. Validation accuracy is the objective.
6. The best configuration is rebuilt and trained.
7. The model is saved as `models/best_nas_model.keras`.
8. Django loads that model for an interactive prediction demo.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/neural_architecture_search.ipynb` and run it from top to bottom.

Then:
```bash
python app.py
```
Open `http://127.0.0.1:8000/`.

The default notebook uses only 8 Optuna trials. Increase this only if your computer has enough CPU time.

## Interview Explanation
> I built a lightweight Neural Architecture Search system with Optuna. Instead of manually deciding the neural-network architecture, I defined a search space covering depth, units, dropout, learning rate, and batch size. Optuna evaluates candidate architectures using validation accuracy and selects the best configuration. I then exposed the selected model through Django so the result can be demonstrated interactively.

## Profiles
GitHub: https://github.com/InfinitePraveen  
LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Dataset
Fashion-MNIST is an open dataset of 28×28 grayscale clothing images with ten classes. The dataset is not committed to the repository.
