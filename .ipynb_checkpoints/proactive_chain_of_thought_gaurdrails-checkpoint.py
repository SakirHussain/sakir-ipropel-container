from typing import Optional, Dict, Any, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaLLM
from guardrails import Guard
from langchain.prompts import PromptTemplate

# Import the filtering function
from student_answer_noncollab_filtering import filter_irrelevant_content

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = OllamaLLM(model="deepseek-r1:32b", temperature=0)

def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    """Computes thematic similarity between student answer and ideal answer using Spacy embeddings."""
    text_emb = sbert_model.encode([student_answer], convert_to_numpy=True)[0]
    q_emb = sbert_model.encode([ideal_answer], convert_to_numpy=True)[0]
    return float(cosine_similarity(text_emb.reshape(1, -1), q_emb.reshape(1, -1))[0][0])

def compute_tfidf_similarity(student_answer: str, ideal_answer: str):
    """Computes TF-IDF similarity between student answer and ideal answer."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_answer, ideal_answer])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

PROCOT_RAIL = """
<rail version="0.1">

<output>
  <object>
    <property name="evaluation_method" type="string"
      description="Evaluation can be Clarification Dialogue, Target-Guided Dialogue, or Non-Collaborative Dialogue, Choose one based on the prompt and explain why and how"/>
    <property name="thought_process" type="string" 
      description="Reasoning before selecting an action."/>
    <property name="action_taken" type="string" 
      description="Chosen action based on evaluation."/>
    <property name="response" type="string" 
      description="Generated feedback with deductions or awards."/>
    <property name="final_adjusted_score" type="number" 
      description="Final adjusted score after refinements."/>
    
    <required name="evaluation_method"/>
    <required name="thought_process"/>
    <required name="action_taken"/>
    <required name="response"/>
    <required name="final_adjusted_score"/>
  </object>
</output>

</rail>
"""

guard = Guard.from_rail_string(PROCOT_RAIL)

def guarded_llm_invoke(prompt: str, debug_label: str) -> Optional[Dict[str, Any]]:
    raw_llm_output = model.invoke(prompt)
#     print(f"\n[RAW LLM OUTPUT - {debug_label}]")
#     print(raw_llm_output)

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        # Instead of storing the entire ValidationOutcome, store just the validated_output dict
        outcome = guard.parse(llm_output=raw_llm_output, messages=messages)
#         print(f"\n[VALIDATED JSON - {debug_label}]")
#         print(outcome.validated_output)  # Show the dict part only
        return outcome.validated_output  # <--- Return the dict
    except Exception as e:
        print(f"\nGuardrails failed to parse JSON for {debug_label}: {e}")
        return None

def generate_structured_eval(
    dialogue_type: str,
    dialogue_desc: str,
    question: str,
    student_answer: str,
    ideal_answer: str,
    rubric: str,
    conversation_history: str,
    available_actions
) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM to produce a valid ProCoTOutput JSON object,
    enforced by Guardrails.
    """

    # If using these similarities only for target-guided, you can conditionally compute them:
    thematic_sim = compute_thematic_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"
    tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"

    prompt_template = PromptTemplate(
        template="""
        You are a professor evaluating a student's answer. 
        Your task is to fairly grade the student's answer and see if the given rubric is answered/met within the answer or not.

        Context and Role:
        - You are responsible for grading fairly and consistently based on the rubric provided.
        - Assign a final_adjusted_score between 0 and max marks as shown in the rubric, where max marks means full credit and 0 means no credit.
        - No assumptions should be made — your evaluation should strictly follow the rubric.
        - Your evaluation method is {dialogue_type}, described below:

        Evaluation Approach:
        {dialogue_desc}

        Evaluation Criteria:
        The following inputs are provided for you to assess the student's response:
        - Question: <question>{question}</question>
        - Student Answer: <student_answer>{student_answer}</student_answer>
        - Ideal Answer: <ideal_answer>{ideal_answer}</ideal_answer>
        - Rubric: <rubric>{rubric}</rubric>
        - Thematic Similarity: {thematic_sim}
        - TF-IDF Similarity: {tfidf_sim}

        Evaluation Framework (Proactive Chain of Thought)
        You must strictly follow the ProCoT framework to ensure structured grading.
        - C (Conversation History): "{conversation_history}"
        - A (Available Actions): {available_actions}

        Scoring Guidelines
        - Any addition or deduction of marks must be explicitly based on whether the rubric is satisfied.
        - Do not assume or add any external information—only infer from the provided inputs.

        Response Format (Strict JSON) :
        ```json
        {{
            "evaluation_method": "How are you choosing to evaluate this answer? Explain the method.",
            "thought_process": "Your reasoning before selecting an action.",
            "action_taken": "Chosen action based on evaluation.",
            "response": "Generated feedback with deductions or awards.",
            "final_adjusted_score": 0.0
        }}
        ```
        No extra text outside JSON.
        """,
        input_variables=[
            "dialogue_type",
            "dialogue_desc",
            "question",
            "student_answer",
            "ideal_answer",
            "rubric",
            "conversation_history",
            "available_actions",
            "thematic_sim",
            "tfidf_sim"
        ],
    )

    formatted_prompt = prompt_template.format(
        dialogue_type=dialogue_type,
        dialogue_desc=dialogue_desc,
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        conversation_history=conversation_history,
        available_actions=available_actions,
        thematic_sim=thematic_sim,
        tfidf_sim=tfidf_sim
    )

    validated_json = guarded_llm_invoke(formatted_prompt, debug_label=dialogue_type)
    return validated_json


def evaluate_answer_by_rubric_items(
    question: str,
    student_answer: str,
    ideal_answer: str,
    rubric_items: List[str]
):
    """
    Evaluates a student's answer *per rubric item*.
    1. First filters the answer to remove irrelevant content.
    2. For each rubric item, runs Clarification Dialogue & Target-Guided Dialogue.
    3. Averages those two scores to get the item score.
    4. Sums all item scores to get the final total.
    
    Returns a structure containing item-by-item details and final total score.
    """

    print("\n--- Starting Per-Rubric-Item Evaluation ---")
    # 1) Filter the student answer once
    filtered_student_answer = filter_irrelevant_content(student_answer, question)
    print("\n--- Ending Per-Rubric-Item Evaluation ---")

    # 2) Common definitions for the dialogues
    conversation_history = ""
    clarification_actions = ["Deduct marks", "Add marks"]
    target_guided_actions = ["Deduct marks", "Add marks"]

    clarification_desc = '''
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.
    '''

    target_guided_desc = '''
    - Determine how many transformations (steps or turns) are needed to thematically convert the student's answer into the ideal answer.
    - Deduct marks based on the necessary transformations.
    - Thematic and TF-IDF similarity are provided.
    '''

    results_by_item = []
    total_score = 0.0

    # 3) Evaluate each rubric item separately
    for idx, rubric_item in enumerate(rubric_items, start=1):
        print(f"\n--- Evaluating Rubric Item #{idx}: {rubric_item} ---")
        
        print("\n--- Starting Clarification Evaluation ---")
        # Clarification Dialogue
        clar_raw = generate_structured_eval(
            "Clarification Dialogue",
            clarification_desc,
            question,
            filtered_student_answer,
            ideal_answer,
            rubric_item,  # Pass just this rubric's text
            conversation_history,
            clarification_actions
        )
        print("\n--- Ending Clarification Evaluation ---")
        
        
        print("\n--- Starting Target guided Evaluation ---")
        # Target-Guided Dialogue
        target_raw = generate_structured_eval(
            "Target-Guided Dialogue",
            target_guided_desc,
            question,
            filtered_student_answer,
            ideal_answer,
            rubric_item,
            conversation_history,
            target_guided_actions
        )
        print("\n--- Ending Target guided Evaluation ---")

        # Extract numeric scores
        if clar_raw and "final_adjusted_score" in clar_raw:
            clar_score = float(clar_raw["final_adjusted_score"])
        else:
            clar_score = 0.0

        if target_raw and "final_adjusted_score" in target_raw:
            target_score = float(target_raw["final_adjusted_score"])
        else:
            target_score = 0.0

        # Average the two
        item_score = (clar_score + target_score) / 2.0
        total_score += item_score

        # Store details
        results_by_item.append({
            "rubric_item": rubric_item,
            "clarification_score": clar_score,
            "target_guided_score": target_score,
            "item_score": item_score,
            "clarification_json": clar_raw,
            "target_guided_json": target_raw
        })
        
        #print(results_by_item)

    # 4) Summarize
    evaluation_result = {
        "total_score": total_score
    }

    return evaluation_result


# (Optional) Keep the older "evaluate_answer" if you still need it
def evaluate_answer(question, student_answer, ideal_answer, rubric):
    """
    Original single-shot approach for reference.
    """
    print("\n--- Starting Evaluation (Legacy) ---")
    print(f"Question: {question}")
    print(f"Student Answer: {student_answer}")
    print(f"Ideal Answer: {ideal_answer}")
    print(f"Rubric: {rubric}")
    
    conversation_history = ""
    clarification_actions = ["Deduct marks", "Add marks"]
    target_guided_actions = ["Deduct marks", "Add marks"]
    
    clarification_desc = '''
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.
    '''
    
    target_guided_desc = '''
    - Determine how many transformations (steps or turns) are needed to thematically convert the student's answer into the ideal answer.
    - Deduct marks based on the necessary transformations.
    - Thematic and TF-IDF similarity are provided.
    '''

    # non collab preprocessing 
    student_answer = filter_irrelevant_content(student_answer, question)
    
    # Clarification Dialogue
    clar_raw = generate_structured_eval(
        "Clarification Dialogue",
        clarification_desc,
        question,
        student_answer,
        ideal_answer,
        rubric,
        conversation_history,
        clarification_actions
    )

    # Target-Guided Dialogue
    target_raw = generate_structured_eval(
        "Target-Guided Dialogue",
        target_guided_desc,
        question,
        student_answer,
        ideal_answer,
        rubric,
        conversation_history,
        target_guided_actions
    )

    clar_score = float(clar_raw["final_adjusted_score"]) if clar_raw and "final_adjusted_score" in clar_raw else 0.0
    target_score = float(target_raw["final_adjusted_score"]) if target_raw and "final_adjusted_score" in target_raw else 0.0

    total_adjusted_score = (clar_score + target_score) / 2.0

    print("\n--- Final Evaluation Summary ---")
    print(f"Clarification Score:   {clar_score}")
    print(f"Target-Guided Score:   {target_score}")
    print(f"Total Combined Score:  {total_adjusted_score:.2f}")

    feedbacks = []
    for res in [clar_raw, target_raw]:
        if res and "response" in res:
            feedbacks.append(res["response"])

    return {
        "total_score": total_adjusted_score,
        "feedback": feedbacks
    }


if __name__ == "__main__":
    question = '''1.7 What is the main difference between a while and a do...while statement?'''

    student_answer = '''1.7 "The â€œdoâ€ statement first evaluates the condition and then executes the lines of code in the statement 0 or more times.<br> The â€œdo whileâ€ statement executes the lines of code and then it evaluates the condition.<br>'''

    ideal_answer = '''1.7 The block inside a do...while statement will execute at least once.'''

    rubric_items = [
        "Accurately identifies that the block inside a do...while statement will execute at least once (2 Marks)",
        "Correctly states the main difference between a while and a do...while statement (3 Marks)"       
    ]

    result = evaluate_answer_by_rubric_items(question, student_answer, ideal_answer, rubric_items)
    print("\n--- Final Result Dictionary ---")
    print(result)
