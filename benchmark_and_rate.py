#!/usr/bin/env python3
import os
from pathlib import Path
import datetime
from benchmark_lod import load_models, load_random_cases, get_case_brief, make_valid_json, extract_output_text
from llm.llm_lod import ClodWrapper
from rate_llm import LLMEvaluator

def run_benchmark_and_rate(models_csv="ai_models_lod.csv", cases_file="random_cases.json"):
    """
    Run benchmarking and rating in a seamless pipeline.
    """
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Prepare CSV file to store benchmark results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_file = f"results/case_model_results_{timestamp}.csv"
    
    # Load models and test cases
    try:
        models = load_models(models_csv)
        print(f"Loaded {len(models)} models")
        
        random_cases = load_random_cases(cases_file)
        print(f"Loaded {len(random_cases)} test cases")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        raise

    # Run benchmarking
    with open(benchmark_file, "w", newline="", encoding='utf-8') as csvfile:
        # Write header
        csvfile.write("Model_ID,Case_Name,Ai_brief,Human_brief\n")

        # Iterate over all models
        for model_config in models:
            try:
                # Initialize wrapper for current model
                llm_wrapper = ClodWrapper(model_config)
                model_info = llm_wrapper.get_model_info()
                model_id = f"{model_info['vendor']}.{model_info['model_name']}"
                
                print(f"\nTesting model: {model_id}")
                
                # Process each case
                for case in random_cases:
                    case_name = case["title"]
                    print(f"Processing case: {case_name}")
                    
                    try:
                        case_brief = get_case_brief(case_name, llm_wrapper)
                        case_brief = case_brief.replace("\n", " ").replace(",", ";")  # Escape commas for CSV
                        
                        # Combine human brief components
                        human_brief = ";".join([
                            f"facts: {case['details']['Facts']}",
                            f"issue: {case['details']['Issue']}",
                            f"decision: {case['details']['Decision']}",
                            f"reasons: {case['details']['Reasons']}",
                            f"ratio: {case['details']['Ratio']}"
                        ])
                        
                        # Write results to CSV
                        csvfile.write(f"{model_id},{case_name},{case_brief},{human_brief}\n")
                        csvfile.flush()  # Ensure data is written immediately
                        
                    except Exception as e:
                        print(f"Error processing case {case_name} with model {model_id}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error initializing model {model_config['model_name']}: {str(e)}")
                continue

    print(f"\nBenchmark results saved to {benchmark_file}")
    
    # Run rating on the benchmark results
    print("\nStarting evaluation of benchmark results...")
    evaluator = LLMEvaluator()
    rated_file = evaluator.evaluate_results(benchmark_file)
    
    print(f"\nComplete pipeline results:")
    print(f"1. Benchmark results: {benchmark_file}")
    print(f"2. Evaluation results: {rated_file}")

if __name__ == "__main__":
    run_benchmark_and_rate()
