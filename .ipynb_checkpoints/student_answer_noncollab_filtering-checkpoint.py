# import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt', quiet=True)

from sentence_transformers import SentenceTransformer
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# 1) Initialize models/pipelines
# print("[DEBUG] Initializing SBERT model and zero-shot classifier...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
# print("[DEBUG] Models loaded successfully!\n")

def sbert_similarity(text: str, question: str) -> float:
    """Compute SBERT-based cosine similarity for (text, question)."""
    if not text.strip():
        return 0.0
    text_emb = sbert_model.encode([text], convert_to_numpy=True)[0]
    q_emb = sbert_model.encode([question], convert_to_numpy=True)[0]
    return float(cosine_similarity(text_emb.reshape(1, -1), q_emb.reshape(1, -1))[0][0])

def tfidf_similarity(text: str, question: str) -> float:
    """Compute TF-IDF-based cosine similarity for (text, question)."""
    if not text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, question])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

def zero_shot_relevance(text: str, question: str) -> float:
    """Return the 'relevant' probability (0-1) using a zero-shot classifier."""
    if not text.strip():
        return 0.0
    zsc_result = zero_shot_classifier(
        sequences=text,
        candidate_labels=["relevant", "irrelevant"],
        hypothesis_template="This text is {} to the question: " + question
    )
    label_scores = dict(zip(zsc_result["labels"], zsc_result["scores"]))
    return float(label_scores.get("relevant", 0.0))

def filter_irrelevant_content(
    student_answer: str,
    question: str,
    window_size=3,
    tolerance=0.034
) -> str:
    """
    Single-pass, Rolling Context approach with Tolerance and Dual Checks.
    
    1) Tokenize student's answer into sentences.
    2) Keep a list of accepted sentences, up to `window_size` for context.
    3) For each new sentence:
       a) recent_context = last N accepted sentences (N=window_size).
       b) We compute old scores for "recent_context" vs question (SBERT/TF-IDF/Zero-Shot).
       c) We form candidate_text = (recent_context + new_sentence).
          Then compute new scores for candidate_text vs question, plus
          single_sent scores for (new_sentence alone vs. question).
       d) If EITHER candidate_text or single_sentence check is >= (old_score - tolerance),
          that method votes "YES."
       e) If >=2 methods vote YES, we accept this sentence (append to accepted_sentences).
          We also update the "old scores" to reflect the newly accepted context.
    4) Return the final accepted text (all accepted sentences joined).
    """

#     print("[DEBUG] Rolling Context + Tolerance Filtering.")
#     print("[DEBUG] Student Answer:\n", student_answer)
#     print("[DEBUG] Question:\n", question)
#     print(f"[DEBUG] window_size={window_size}, tolerance={tolerance}")

    sentences = sent_tokenize(student_answer)
#     print(f"[DEBUG] Found {len(sentences)} sentences.")

    accepted_sentences = []
    # We'll track the "old" scores from the last accepted context block
    # Start with no context => 0.0 for old_sbert/old_tfidf/old_zsc
    old_sbert = 0.0
    old_tfidf = 0.0
    old_zsc = 0.0

    for idx, sent in enumerate(sentences):
        sent_str = sent.strip()
        if not sent_str:
            continue

#         print(f"\n[DEBUG] Sentence {idx+1}: {sent_str}")
        # Build the rolling context
        recent_context = accepted_sentences[-window_size:]
        context_text = " ".join(recent_context)

        # 1) Scores for the old context (just so we can see them)
        #    Actually we already hold them as old_sbert/old_tfidf/old_zsc,
        #    but let's do it explicitly if you want more debugging info:
        # old_sbert = sbert_similarity(context_text, question)
        # old_tfidf = tfidf_similarity(context_text, question)
        # old_zsc = zero_shot_relevance(context_text, question)
        # We'll rely on the stored old scores instead.

        # 2) Candidate text = context + this new sentence
        if context_text.strip():
            candidate_text = context_text + " " + sent_str
        else:
            candidate_text = sent_str

        # 3) Evaluate new context
        cand_sbert = sbert_similarity(candidate_text, question)
        cand_tfidf = tfidf_similarity(candidate_text, question)
        cand_zsc = zero_shot_relevance(candidate_text, question)

        # 4) Evaluate single sentence alone
        single_sbert = sbert_similarity(sent_str, question)
        single_tfidf = tfidf_similarity(sent_str, question)
        single_zsc = zero_shot_relevance(sent_str, question)

#         print("[DEBUG] Old scores => SBERT: {:.4f}, TF-IDF: {:.4f}, ZSC: {:.4f}".format(
#             old_sbert, old_tfidf, old_zsc
#         ))
#         print("[DEBUG] Candidate ctx => SBERT: {:.4f}, TF-IDF: {:.4f}, ZSC: {:.4f}".format(
#             cand_sbert, cand_tfidf, cand_zsc
#         ))
#         print("[DEBUG] Single sent => SBERT: {:.4f}, TF-IDF: {:.4f}, ZSC: {:.4f}".format(
#             single_sbert, single_tfidf, single_zsc
#         ))

        # 5) Tolerant checking:
        # Method votes "YES" if either candidate_text or single_sentence
        # is at least old_score - tolerance
        sbert_vote = (
            (cand_sbert + tolerance >= old_sbert) or
            (single_sbert + tolerance >= old_sbert)
        )
        tfidf_vote = (
            (cand_tfidf + tolerance >= old_tfidf) or
            (single_tfidf + tolerance >= old_tfidf)
        )
        zsc_vote = (
            (cand_zsc + tolerance >= old_zsc) or
            (single_zsc + tolerance >= old_zsc)
        )

        votes = sum([sbert_vote, tfidf_vote, zsc_vote])
#         print(f"[DEBUG] Votes => SBERT: {sbert_vote}, TF-IDF: {tfidf_vote}, ZSC: {zsc_vote} (Total={votes})")

        # 6) Majority: 2 of 3
        if votes >= 2:
#             print("[DEBUG] Accepting this sentence.")
            accepted_sentences.append(sent_str)
            # Update the old scores to reflect new context
            # We consider the candidate_text the new context
            old_sbert = cand_sbert
            old_tfidf = cand_tfidf
            old_zsc = cand_zsc
#         else:
#             print("[DEBUG] Rejecting this sentence.")

    # Reconstruct final accepted answer
    final_text = " ".join(accepted_sentences)
#     print("\n[DEBUG] FINAL ACCEPTED TEXT:")
    print(final_text)
    return final_text


# if __name__ == "__main__":
#     # Example usage
#     question = '''
#     The city council plans to develop a mobile app to enhance urban mobility by providing residents with information on public transport, bike-sharing, and ride-hailing options. Due to changing transportation policies and user needs, the app’s requirements are evolving. With a limited budget and the need for a quick release, the council aims to roll out features in phases, starting with essential transport information and later adding real-time updates and payment integration.
#     a. How will you implement the Agile process model for the above scenario ? (5 Marks)
#     b. Discuss how eXtreme Programming (XP) can support the development of the mobile app.(5 Marks)
#     '''
    
#     student_answer = """
#         Part(a) Agile is philosophy that revolves around agility in software development and customer satisfaction.
#         It involves integrating the customer to be a part of the development team in order to recueve quick feedback and fast implementations.
#         In the case of a mobile application in improve urban mobility, we will rely on building the application in increments. This will require the application to have high modularity.
#         The modules can be as follows : bikesharing, ride hailing, proximity radar, ride selection/scheduling. But i love having pizzas on a wednesday afternoon which be pivtol in this case as well.
#         The bike sharing and ride hailing modules are mainly UI based and can be developed in one sprint. The feedback can be obtained from a select group of citizens or lauch a test application in beta state to all phones.
#         The core logic - proximity radar, to define how close or far awat te application must look for a ride and ride selection is all about selecting a ride for the user without clashing with other users.
#         This is developed in subsequent sprint cycles and can be tested by limited area lauch to citizens to bring out all the runtime errors and bugs. Addtionally Agile is all about speed and i want more speed.

#         Part(b) eXtreme progreamming relies on maily very fast development and mazimizing customer satisfaction.
#         Since quick release is important along with subsequent rollouts this is a good SDLC model.
#         The plannig is the first phase of the SDLC model. Here the requirements, need not be rigid or well defined or even formally defined. The requirements are communicated roughly and the production can begin. Here a ride application with public transport, bike sharing and ride hailing.
#         Based on this alone, the architecture/software architecture can be obtained.
#         Once the software architecture is defined for the interation, the coding/implementation begins.
#         Coding is usually pair programming. The modules selected such as UI, bikesharing, ride hailing and public transport are developed.
#         Once they are developed, they are tested agasint the member of the team or in this case a public jury/citizen jury is used to check the appeal of the UI.
#         If it is satisfactory, the component is completed and implemented into the application, if not, the feedback is sent as an input for the next iteration and the process is repeated again.
    
#     """

#     print("[DEBUG] Example question:\n", question)
#     print("[DEBUG] Example answer:\n", student_answer)

#     filtered_text = filter_irrelevant_content(student_answer, question)
#     print("\n[DEBUG] FINAL OUTPUT:")
#     print(filtered_text)
