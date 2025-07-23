import json
import sys
import os
from typing import Optional

import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, ENV_VARIABLES
from agents.evaluation import DataLoader, Evaluator
from agents.rag.base import RAGAgent

async def main(prediction_run: Optional[str] = None):
    # Load the data
    # test_data = DATA_DIR / "data_to_process" / "test.jsonl"
    # data_loader = DataLoader(file_dir=test_data)

    validation_data = DATA_DIR / "golden_dataset.json"
    data_loader = DataLoader(file_dir=validation_data)

    # Initialize the assistant
    assistant = RAGAgent(llm_provider="azure", retrieval_args={})

    # Initialize the evaluator
    evaluator = Evaluator(dataloader=data_loader, model_pipeline=assistant)

    evaluations_run_dir = DATA_DIR / "evaluations"
    if prediction_run is None:
        # Run the evaluation
        sampled_data = await evaluator.run_prediction(size_sample=300)

        if not evaluations_run_dir.exists():
            evaluations_run_dir.mkdir(parents=True, exist_ok=True)
        # list runs folders
        runs_folders = evaluations_run_dir.glob("run_*")
        run_id = len(list(runs_folders)) + 1
        run_dir = evaluations_run_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # save the results
        with open(run_dir / "predictions_results.json", "w", encoding="utf-8") as f:
            json.dump(sampled_data, f, indent=4)

    else: #load the results
        run_dir = evaluations_run_dir / prediction_run
        with open(run_dir / "predictions_results.json", "r", encoding="utf-8") as f:
            sampled_data = json.load(f)

    # Run the evaluation
    sampled_data = await evaluator.evaluate_prediction(sampled_data)

    # save the results
    with open(run_dir / "evaluations_results.json", "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=4)

async def get_metrics(prediction_run: str):
    evaluations_run_dir = DATA_DIR / "evaluations"
    run_dir = evaluations_run_dir / prediction_run
    with open(run_dir / "evaluations_results.json", "r", encoding="utf-8") as f:
        sampled_data = json.load(f)

        # Extraer los scores
    scores = [item["evaluation"]["score"] for item in sampled_data]

    # Contar ocurrencias en el orden deseado
    score_order = ["none", "few", "most", "all"]
    counter = Counter(scores)
    score_counts = [counter.get(score, 0) for score in score_order]

    # Graficar
    fig, ax = plt.subplots()
    bars = ax.bar(score_order, score_counts, color="#4a90e2", edgecolor="black")

    # Añadir etiquetas en la parte superior de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Estilo del gráfico
    ax.set_xlabel("Score")
    ax.set_ylabel("Cantidad")
    ax.set_title("Distribución de Scores de Evaluación")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(run_dir / "distribucion_scores.png", dpi=300, bbox_inches='tight')

    # Gráfico de torta con porcentajes
    fig, ax = plt.subplots()

    # Valores y etiquetas
    labels = score_order
    sizes = score_counts
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]  # Puedes ajustar los colores

    # Crear gráfico de torta
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops=dict(color="white"),
        wedgeprops=dict(edgecolor='black')
    )

    # Estilo
    ax.axis('equal')  # Mantener aspecto de círculo perfecto
    plt.title("Distribución de Scores (%)")

    # Guardar gráfico de torta
    plt.savefig(run_dir / "distribucion_scores_pie.png", dpi=300, bbox_inches='tight')
    

# Run the main function
if __name__ == "__main__":
    import asyncio
    #asyncio.run(main("run_1"))
    asyncio.run(get_metrics("run_1"))