import pandas as pd
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

class EmbeddingEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if pd.isna(text1) or pd.isna(text2):
            return None
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()

    def evaluate_results(self, input_file: str) -> str:
        """Evaluate section-wise similarity from a CSV file."""
        df = pd.read_csv(input_file)

        sections = ['facts', 'issue', 'decision', 'reasons', 'ratio']
        for section in sections:
            col_name = f'{section}_similarity'
            df[col_name] = [
                self.compute_similarity(row[f'ai_{section}'], row[f'human_{section}'])
                for _, row in df.iterrows()
            ]
            print(f"Finished computing {col_name}")

        # Save results
        Path("results").mkdir(exist_ok=True)
        output_file = 'results/evaluated_case_model_results_with_section_similarity.csv'
        df.to_csv(output_file, index=False)
        print(f"Evaluation complete. Results saved to {output_file}")
        return output_file

def main(input_file: str):
    evaluator = EmbeddingEvaluator()
    return evaluator.evaluate_results(input_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rate_llm.py <input_csv_file>")
        sys.exit(1)
    main(sys.argv[1])
