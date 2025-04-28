import json
import os
import numpy as np
import scipy.stats as stats
from proactive_chain_of_thought_gaurdrails import evaluate_answer_by_rubric_items

# Path to dataset
DATASET_FILE = "cleaned_dataset.json"
OUTPUT_RESULTS_FILE = "procot_evaluation_results.json"
PROGRESS_FOLDER = "test_outputs"

# Ensure the progress folder exists
os.makedirs(PROGRESS_FOLDER, exist_ok=True)

# Load dataset
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Function to call ProCoT evaluation
def generate_structured_eval(question, student_answer, ideal_answer, rubric):
    """
    Calls the ProCoT evaluation function to get a structured score.
    """
    score_result = evaluate_answer_by_rubric_items(question, student_answer, ideal_answer, rubric)
    return score_result.get("total_score", 0.0)  # Ensure we return a float score

# Function to compute percentage error distribution
def compute_error_distribution(errors):
    """
    Returns a dictionary showing percentage of errors in different brackets.
    """
    error_ranges = {
        "0-1 marks": 0,
        "1-2 marks": 0,
        "2-3 marks": 0,
        "3-4 marks": 0,
        "4+ marks": 0
    }

    for error in errors:
        if error < 1:
            error_ranges["0-1 marks"] += 1
        elif error < 2:
            error_ranges["1-2 marks"] += 1
        elif error < 3:
            error_ranges["2-3 marks"] += 1
        elif error < 4:
            error_ranges["3-4 marks"] += 1
        else:
            error_ranges["4+ marks"] += 1

    # Convert to percentages
    total = len(errors)
    return {k: (v / total) * 100 for k, v in error_ranges.items()} if total > 0 else error_ranges

# Store all evaluation results
results = []
human_avg_scores = []
procot_scores = []

# Counter to limit execution
eval_count = 0
MAX_EVALUATIONS = 1000  # Limit to 1000 evaluations
SAVE_INTERVAL = 25  # Save every 25 evaluations

flag = 0

# Iterate over the entire dataset
for entry in dataset:
    question_text = entry["question"]
    ideal_answer = entry["ideal_answer"]
    rubric = entry["rubric"]

    # Evaluate each student answer
    for student in entry["student_answers"]:        
        print(f"----- PERFORMING == {eval_count + 1}")

        if eval_count >= MAX_EVALUATIONS:
            flag = 1
            break  # Stop after 1000 evaluations
        
        student_id = student["student_id"]
        student_answer = student["answer"]
        human_scores = student["human_scores"]
        human_avg_score = np.mean(human_scores)  # Average human score

        # Run ProCoT evaluation
        procot_score = generate_structured_eval(question_text, student_answer, ideal_answer, rubric)

        # Store results
        results.append({
            "question": question_text,
            "student_id": student_id,
            "student_answer": student_answer,
            "human_scores": human_scores,
            "human_avg_score": human_avg_score,
            "procot_score": procot_score
        })

        # Store scores for statistical analysis
        human_avg_scores.append(human_avg_score)
        procot_scores.append(procot_score)
        
        eval_count += 1  

        # Every 25 evaluations, save progress
        if eval_count % SAVE_INTERVAL == 0:
            # Convert to numpy arrays for stats calculations
            np_human_avg_scores = np.array(human_avg_scores)
            np_procot_scores = np.array(procot_scores)
            np_absolute_errors = np.abs(np_human_avg_scores - np_procot_scores)

            # Compute ML metrics
            mae = np.mean(np_absolute_errors)  # Mean Absolute Error
            mse = np.mean((np_human_avg_scores - np_procot_scores) ** 2)  # Mean Squared Error
            correlation = stats.pearsonr(np_human_avg_scores, np_procot_scores)[0] if len(np_human_avg_scores) > 1 else None

            # Compute error distribution
            error_distribution = compute_error_distribution(np_absolute_errors)
            
            # Compute overall error percentage
            overall_error_percentage = np.mean(np_absolute_errors > 0) * 100  # % of evaluations where ProCoT made any error

            # Save progress to JSON file
            progress_file = os.path.join(PROGRESS_FOLDER, f"progress_{eval_count}.json")
            progress_data = {
                "evaluations_done": eval_count,
                "MAE": mae,
                "MSE": mse,
                "Pearson_Correlation": correlation,
                "Overall_Error_Percentage": overall_error_percentage,
                "Error_Distribution": error_distribution,
                "recent_25_results": results[-25:]  # Store last 25 results for debugging
            }

            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=4, ensure_ascii=False)

            print(f"\n📌 Progress saved after {eval_count} evaluations in {progress_file}")

    if flag == 1:
        break

# Final Metrics Calculation
human_avg_scores = np.array(human_avg_scores)
procot_scores = np.array(procot_scores)
absolute_errors = np.abs(human_avg_scores - procot_scores)

# Compute Final ML metrics
mae = np.mean(absolute_errors)  # Mean Absolute Error
mse = np.mean((human_avg_scores - procot_scores) ** 2)  # Mean Squared Error
correlation = stats.pearsonr(human_avg_scores, procot_scores)[0] if len(human_avg_scores) > 1 else None
error_distribution = compute_error_distribution(absolute_errors)

# Print Final Evaluation Results
print("\n🔍 Final Evaluation Results for Entire Dataset\n")
print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📊 Mean Squared Error (MSE): {mse:.4f}")
print(f"📊 Pearson Correlation Coefficient: {correlation:.4f}" if correlation is not None else "📊 Pearson Correlation Coefficient: Not enough data points")
print(f"📊 Final Error Distribution: {error_distribution}")

# Save final results
with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Final results saved to {OUTPUT_RESULTS_FILE}")
