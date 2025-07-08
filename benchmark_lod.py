import json
import datetime
import csv
import os
from time import sleep
from typing import Dict, List
from pathlib import Path
from llm.llm_wrapper import LLM_Wrapper
from llm.llm_lod import ClodWrapper
import re

def load_models(csv_path: str = "ai_models_lod.csv") -> List[Dict]:
    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_config = {
                "model_name": row['Model Name'],
                "endpoint": "https://api.clod.io/v1",
                "API_key": os.environ.get('CLOD_API_KEY'),
                "models_csv_path": csv_path,
                "provider": {
                    "name": row['Provider Name'],
                    "base_url": row['Provider Base URL']
                }
            }
            models.append(model_config)
    return models

def get_case_brief(case_name: str, llm_wrapper: LLM_Wrapper) -> Dict[str, str]:
    prompt = f"Human: Provide a case brief for {case_name} with only facts, issue, decision, reasons, and ratio, formatted strictly as JSON: " + \
             '{"facts":"","issue":"","decision":"","reasons":"","ratio":""}. No additional text, just JSON. If unknown, respond with \"I don\'t know\" only. Assistant:'

    response = llm_wrapper.invoke(prompt)
    cleaned = make_valid_json(response)
    try:
        parsed = json.loads(cleaned)
        return {
            "ai_facts": parsed.get("facts", ""),
            "ai_issue": parsed.get("issue", ""),
            "ai_decision": parsed.get("decision", ""),
            "ai_reasons": parsed.get("reasons", ""),
            "ai_ratio": parsed.get("ratio", "")
        }
    except Exception as e:
        return {
            "ai_facts": "ERROR",
            "ai_issue": "ERROR",
            "ai_decision": "ERROR",
            "ai_reasons": "ERROR",
            "ai_ratio": "ERROR"
        }

def make_valid_json(input_str: str) -> str:
    """
    Extract the first valid JSON object (dict) from the input string.
    Strips out extra characters, markdown code fences, etc.
    """
    # Remove Markdown code blocks if present
    input_str = re.sub(r"```(?:json)?", "", input_str).strip()

    # Use a stack-based approach to find the first complete JSON object
    stack = []
    start_index = None
    for i, char in enumerate(input_str):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index is not None:
                    json_str = input_str[start_index:i+1]
                    return json_str

    raise ValueError("No valid JSON object found in the input string.")

def load_random_cases(file_path: str = "random_cases.json") -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
        return json.load(f)

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_name = f"results/case_model_results_{timestamp}.csv"

    models = load_models()
    try:
        random_cases = load_random_cases()
        print(f"Loaded {len(random_cases)} test cases")
    except Exception as e:
        print(f"Error loading test cases: {str(e)}")
        raise

    with open(csv_file_name, "w", newline="", encoding='utf-8') as csvfile:
        fieldnames = [
            "Model_ID", "Case_Name",
            "ai_facts", "ai_issue", "ai_decision", "ai_reasons", "ai_ratio",
            "human_facts", "human_issue", "human_decision", "human_reasons", "human_ratio"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_config in models:
            try:
                llm_wrapper = ClodWrapper(model_config)
                model_info = llm_wrapper.get_model_info()
                model_id = f"{model_info['vendor']}.{model_info['model_name']}"
                print(f"\nTesting model: {model_id}")

                for case in random_cases:
                    case_name = case["title"]
                    print(f"Processing case: {case_name}")
                    try:
                        ai_output = get_case_brief(case_name, llm_wrapper)
                        human_details = case['details']

                        writer.writerow({
                            "Model_ID": model_id,
                            "Case_Name": case_name,
                            "ai_facts": ai_output["ai_facts"],
                            "ai_issue": ai_output["ai_issue"],
                            "ai_decision": ai_output["ai_decision"],
                            "ai_reasons": ai_output["ai_reasons"],
                            "ai_ratio": ai_output["ai_ratio"],
                            "human_facts": human_details["Facts"],
                            "human_issue": human_details["Issue"],
                            "human_decision": human_details["Decision"],
                            "human_reasons": human_details["Reasons"],
                            "human_ratio": human_details["Ratio"]
                        })
                        sleep(5)
                    except Exception as e:
                        print(f"Error processing case {case_name} with model {model_id}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error initializing model {model_config['model_name']}: {str(e)}")
                continue
    print(f"\nResults saved to {csv_file_name}")
