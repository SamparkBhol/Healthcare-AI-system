import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class VisualizationTools:
    def __init__(self):
        pass

    def plot_agent_decision(self, decisions, labels):
        plt.figure(figsize=(10, 5))
        plt.plot(decisions, label="Agent Decisions", color="blue")
        plt.plot(labels, label="True Labels", color="red")
        plt.xlabel("Time Step")
        plt.ylabel("Decision Value")
        plt.title("Agent Decisions Over Time")
        plt.legend()
        plt.show()

    def plot_model_performance(self, metrics):
        plt.figure(figsize=(8, 6))
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.bar(names, values, color='green')
        plt.title("Model Performance Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.show()

    def visualize_causal_graph(self, adjacency_matrix, node_labels):
        G = nx.from_numpy_matrix(adjacency_matrix)
        pos = nx.spring_layout(G)

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, labels={i: node_labels[i] for i in range(len(node_labels))})
        plt.title("Causal Graph Visualization")
        plt.show()

if __name__ == "__main__":
    viz_tools = VisualizationTools()

    # Example for agent decisions
    decisions = np.random.rand(100)
    true_labels = np.random.randint(2, size=100)
    viz_tools.plot_agent_decision(decisions, true_labels)

    # Example for model performance
    performance_metrics = {"accuracy": 0.85, "precision": 0.78, "recall": 0.82, "f1_score": 0.80}
    viz_tools.plot_model_performance(performance_metrics)

    # Example for causal graph
    adjacency_matrix = np.random.randint(2, size=(5, 5))
    node_labels = ["Symptom", "Diagnosis", "Treatment", "Outcome", "Side Effect"]
    viz_tools.visualize_causal_graph(adjacency_matrix, node_labels)
