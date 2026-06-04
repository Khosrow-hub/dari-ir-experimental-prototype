import re
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.contingency_tables import mcnemar

CORPUS_PATH = "dari_clean_corpus.csv"
QUERIES_PATH = "dari_evaluation_queries_150.csv"
TOP_K = 5


def normalize_dari(text):
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
        "‌": " ",
        "\u200f": "",
        "\u200e": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    return text

    
def expand_query(query):
    query = normalize_dari(query)

    expansion_map = {
        "فناوری": "تکنالوژی",
        "محرمانه": "محرم",
        "محرمانه پنتاگون": "محرم پنتاگون",
        "پهپادی": "طیاره بی سرنشین",
        "پهپاد": "طیاره بی سرنشین",
        "موشکی": "میزایلی",
        "موشک": "میزایل",
        "اتمی": "هسته ای",
        "سامانه": "سیستم",
        "ذخایر آب": "منابع آب",
        "آب نوشیدنی": "آب آشامیدنی",
        "کمبود آب": "بحران آب",
        "افزایش خشکسالی": "شیوع خشکسالی",
        "نشانه های": "علائم",
        "مالیات": "عوارض",
        "فروش تسلیحاتی": "فروش تسلیحات",
        "توافق دفاعی": "همکاری دفاعی",
        "رئیس جمهور چین": "شی جین پینگ",
        "دیدار رسمی": "سفر رسمی",
        "خرید و فروش طلا": "واردات طلا",
        "آفت گندم": "سن گندم",
        "بیماری هانتا": "ویروس هانتا",
        "گسترش ایبولا": "شیوع بیماری ایبولا",
        "تغییر آب و هوا": "تغییرات اقلیمی",
    }

    expanded = query

    for source, target in expansion_map.items():
        if source in expanded:
            expanded = expanded.replace(source, target)

    return expanded        

    text = str(text)

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
        "‌": " ",
        "\u200f": "",
        "\u200e": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_trigrams(text):
    text = normalize_dari(text).replace(" ", "")
    if len(text) < 3:
        return {text}
    return {text[i:i+3] for i in range(len(text) - 2)}


def trigram_similarity(query, document):
    q_grams = char_trigrams(query)
    d_grams = char_trigrams(document)

    if not q_grams or not d_grams:
        return 0

    intersection = len(q_grams.intersection(d_grams))
    union = len(q_grams.union(d_grams))

    return (intersection / union) * 100


def exact_search(query, corpus_df, top_k=TOP_K):
    query_norm = normalize_dari(query)
    results = []

    for _, row in corpus_df.iterrows():
        if query_norm in row["normalized_text"]:
            results.append({
                "id": row["id"],
                "text": row["text"],
                "score": 100
            })

    return pd.DataFrame(results).head(top_k)


def fuzzy_search(query, corpus_df, top_k=TOP_K):
    query_norm = normalize_dari(query)
    results = []

    for _, row in corpus_df.iterrows():
        score = fuzz.partial_ratio(query_norm, row["normalized_text"])
        results.append({
            "id": row["id"],
            "text": row["text"],
            "fuzzy_score": round(score, 2)
        })

    return pd.DataFrame(results).sort_values(by="fuzzy_score", ascending=False).head(top_k)


def trigram_search(query, corpus_df, top_k=TOP_K):
    query_norm = normalize_dari(query)
    results = []

    for _, row in corpus_df.iterrows():
        score = trigram_similarity(query_norm, row["normalized_text"])
        results.append({
            "id": row["id"],
            "text": row["text"],
            "trigram_score": round(score, 2)
        })

    return pd.DataFrame(results).sort_values(by="trigram_score", ascending=False).head(top_k)


def hybrid_search(query, corpus_df, top_k=TOP_K, fuzzy_weight=0.6, trigram_weight=0.4):
    query_norm = normalize_dari(query)
    results = []

    for _, row in corpus_df.iterrows():
        fuzzy_score = fuzz.partial_ratio(query_norm, row["normalized_text"])
        trigram_score = trigram_similarity(query_norm, row["normalized_text"])
        final_score = (fuzzy_weight * fuzzy_score) + (trigram_weight * trigram_score)

        results.append({
            "id": row["id"],
            "text": row["text"],
            "fuzzy_score": round(fuzzy_score, 2),
            "trigram_score": round(trigram_score, 2),
            "final_score": round(final_score, 2)
        })

    return pd.DataFrame(results).sort_values(by="final_score", ascending=False).head(top_k)

def expanded_hybrid_search(query, corpus_df, top_k=TOP_K, fuzzy_weight=0.5, trigram_weight=0.5):
    expanded_query = expand_query(query)

    if expanded_query is None or str(expanded_query).strip() == "":
        expanded_query = query

    return hybrid_search(
        expanded_query,
        corpus_df,
        top_k,
        fuzzy_weight=fuzzy_weight,
        trigram_weight=trigram_weight
    )


def build_tfidf_model(corpus_df):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(corpus_df["normalized_text"])
    return vectorizer, tfidf_matrix


def tfidf_search(query, corpus_df, vectorizer, tfidf_matrix, top_k=TOP_K):
    query_norm = normalize_dari(query)

    if query_norm is None:
        query_norm = ""

    query_norm = str(query_norm)

    query_vector = vectorizer.transform([query_norm])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = corpus_df.iloc[idx]
        results.append({
            "id": row["id"],
            "text": row["text"],
            "tfidf_score": round(similarities[idx] * 100, 2)
        })

    return pd.DataFrame(results)

def enhanced_hybrid_search(
    query,
    corpus_df,
    vectorizer,
    tfidf_matrix,
    top_k=TOP_K,
    fuzzy_weight=0.45,
    tfidf_weight=0.40,
    trigram_weight=0.15
):
    query_norm = normalize_dari(query)

    query_vector = vectorizer.transform([query_norm])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten() * 100

    results = []

    for idx, row in corpus_df.iterrows():
        text_norm = row["normalized_text"]

        fuzzy_score = fuzz.WRatio(query_norm, text_norm)
        trigram_score = trigram_similarity(query_norm, text_norm)
        tfidf_score = tfidf_scores[idx]

        final_score = (
            fuzzy_weight * fuzzy_score
            + tfidf_weight * tfidf_score
            + trigram_weight * trigram_score
        )

        results.append({
            "id": row["id"],
            "text": row["text"],
            "fuzzy_wratio": round(fuzzy_score, 2),
            "char_tfidf_score": round(tfidf_score, 2),
            "trigram_score": round(trigram_score, 2),
            "final_score": round(final_score, 2)
        })

    return pd.DataFrame(results).sort_values(
        by="final_score",
        ascending=False
    ).head(top_k)


def evaluate_method(search_function, corpus_df, queries_df, method_name, top_k=TOP_K):
    rows = []

    for _, qrow in queries_df.iterrows():
        query = qrow["query_text"]
        relevant_id = int(qrow["relevant_record_id"])

        results = search_function(query, corpus_df, top_k)
        retrieved_ids = results["id"].tolist() if not results.empty else []

        hit = 1 if relevant_id in retrieved_ids else 0

        precision_at_k = hit / top_k
        recall_at_k = hit

        if precision_at_k + recall_at_k == 0:
            f1_at_k = 0
        else:
            f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

        rows.append({
            "method": method_name,
            "query_id": qrow["query_id"],
            "query_type": qrow["query_type"],
            "query_text": query,
            "relevant_record_id": relevant_id,
            "hit@5": hit,
            "precision@5": round(precision_at_k, 4),
            "recall@5": round(recall_at_k, 4),
            "f1@5": round(f1_at_k, 4),
            "retrieved_ids": retrieved_ids
        })

    return pd.DataFrame(rows)

def evaluate_all_methods(corpus_df, queries_df, vectorizer, tfidf_matrix):
    exact_eval = evaluate_method(
        lambda q, df, k: exact_search(q, df, k),
        corpus_df,
        queries_df,
        "Exact Matching"
    )

    tfidf_eval = evaluate_method(
        lambda q, df, k: tfidf_search(q, df, vectorizer, tfidf_matrix, k),
        corpus_df,
        queries_df,
        "TF-IDF"
    )

    fuzzy_eval = evaluate_method(
        lambda q, df, k: fuzzy_search(q, df, k),
        corpus_df,
        queries_df,
        "Fuzzy Only"
    )

    trigram_eval = evaluate_method(
        lambda q, df, k: trigram_search(q, df, k),
        corpus_df,
        queries_df,
        "Trigram Only"
    )

    hybrid_eval = evaluate_method(
        lambda q, df, k: hybrid_search(q, df, k),
        corpus_df,
        queries_df,
        "Hybrid Fuzzy + Trigram"
    )

    expanded_hybrid_eval = evaluate_method(
        lambda q, df, k: expanded_hybrid_search(q, df, k),
        corpus_df,
        queries_df,
        "Expanded Hybrid"
    )

    enhanced_hybrid_eval = evaluate_method(
        lambda q, df, k: enhanced_hybrid_search(q, df, vectorizer, tfidf_matrix, k),
        corpus_df,
        queries_df,
        "Enhanced Hybrid"
    )



    all_eval = pd.concat(
        [    exact_eval,
    tfidf_eval,
    fuzzy_eval,
    trigram_eval,
    hybrid_eval,
    expanded_hybrid_eval],
        ignore_index=True
    )

    return all_eval


def summarize_results(all_eval):
    overall = (
        all_eval
        .groupby("method")[["hit@5", "precision@5", "recall@5", "f1@5"]]
        .mean()
        .reset_index()
    )

    overall["top_5_accuracy"] = (overall["hit@5"] * 100).round(2)
    overall["precision@5"] = overall["precision@5"].round(4)
    overall["recall@5"] = overall["recall@5"].round(4)
    overall["f1@5"] = overall["f1@5"].round(4)

    overall = overall.drop(columns=["hit@5"])

    by_type = (
        all_eval
        .groupby(["method", "query_type"])[["hit@5", "precision@5", "recall@5", "f1@5"]]
        .mean()
        .reset_index()
    )

    by_type["top_5_accuracy"] = (by_type["hit@5"] * 100).round(2)
    by_type["precision@5"] = by_type["precision@5"].round(4)
    by_type["recall@5"] = by_type["recall@5"].round(4)
    by_type["f1@5"] = by_type["f1@5"].round(4)

    by_type = by_type.drop(columns=["hit@5"])

    return overall, by_type

def run_weight_tuning(corpus_df, queries_df):
    weight_configs = [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.4, 0.6),
    ]

    results = []

    for fw, tw in weight_configs:
        eval_rows = []

        for _, qrow in queries_df.iterrows():
            query = qrow["query_text"]
            relevant_id = int(qrow["relevant_record_id"])

            res = hybrid_search(
                query,
                corpus_df,
                TOP_K,
                fuzzy_weight=fw,
                trigram_weight=tw
            )

            retrieved_ids = res["id"].tolist()
            hit = 1 if relevant_id in retrieved_ids else 0

            eval_rows.append(hit)

        accuracy = (sum(eval_rows) / len(eval_rows)) * 100

        results.append({
            "fuzzy_weight": fw,
            "trigram_weight": tw,
            "top_5_accuracy": round(accuracy, 2)
        })

    return pd.DataFrame(results)    

def run_mcnemar_test(all_eval, method_a="TF-IDF", method_b="Expanded Hybrid"):
    method_a_df = all_eval[all_eval["method"] == method_a].sort_values("query_id")
    method_b_df = all_eval[all_eval["method"] == method_b].sort_values("query_id")

    both_correct = 0
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    both_wrong = 0

    for (_, row_a), (_, row_b) in zip(method_a_df.iterrows(), method_b_df.iterrows()):
        a_hit = row_a["hit@5"]
        b_hit = row_b["hit@5"]

        if a_hit == 1 and b_hit == 1:
            both_correct += 1
        elif a_hit == 1 and b_hit == 0:
            a_correct_b_wrong += 1
        elif a_hit == 0 and b_hit == 1:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1

    contingency_table = [
        [both_correct, a_correct_b_wrong],
        [a_wrong_b_correct, both_wrong]
    ]

    result = mcnemar(contingency_table, exact=True)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "p_value": result.pvalue
    }    


st.set_page_config(
    page_title="Dari Text Search Prototype",
    layout="wide"
)

st.title("Dari Text Search Prototype")
st.write("Experimental comparison of Exact Matching, TF-IDF, Fuzzy Matching, Character Trigram, and Hybrid Retrieval.")

try:
    corpus_df = pd.read_csv(CORPUS_PATH)
    queries_df = pd.read_csv(QUERIES_PATH)

    corpus_df["text"] = corpus_df["text"].fillna("").astype(str)
    corpus_df["normalized_text"] = corpus_df["text"].apply(normalize_dari).fillna("").astype(str)
    queries_df["query_text"] = queries_df["query_text"].fillna("").astype(str)

    vectorizer, tfidf_matrix = build_tfidf_model(corpus_df)

    st.success(f"Corpus loaded: {len(corpus_df)} records")
    st.success(f"Evaluation queries loaded: {len(queries_df)} queries")

    st.divider()

    st.header("Search Engine Demo")

    query = st.text_input("Enter a Dari query:", value="بحران اب در افغانستن")
    top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=5)

    if st.button("Search"):
        tabs = st.tabs([
            "Exact",
            "TF-IDF",
            "Fuzzy Only",
            "Trigram Only",
            "Hybrid",
            "Enhanced Hybrid"
        ])

        with tabs[0]:
            st.subheader("Exact Keyword Matching")
            res = exact_search(query, corpus_df, top_k)
            if res.empty:
                st.warning("No exact match found.")
            else:
                st.dataframe(res, use_container_width=True)

        with tabs[1]:
            st.subheader("TF-IDF Baseline")
            res = tfidf_search(query, corpus_df, vectorizer, tfidf_matrix, top_k)
            st.dataframe(res, use_container_width=True)

        with tabs[2]:
            st.subheader("Fuzzy Only")
            res = fuzzy_search(query, corpus_df, top_k)
            st.dataframe(res, use_container_width=True)

        with tabs[3]:
            st.subheader("Character Trigram Only")
            res = trigram_search(query, corpus_df, top_k)
            st.dataframe(res, use_container_width=True)

        with tabs[4]:
            st.subheader("Hybrid Fuzzy + Character Trigram")
            res = hybrid_search(query, corpus_df, top_k)
            st.dataframe(res, use_container_width=True)
        
        with tabs[5]:
            st.subheader("Enhanced Hybrid Retrieval")
            res = enhanced_hybrid_search(query, corpus_df, vectorizer, tfidf_matrix, top_k)
            st.dataframe(res, use_container_width=True)

    st.divider()

    st.header("Automatic Evaluation")

    if st.button("Run Full Evaluation"):
        all_eval = evaluate_all_methods(corpus_df, queries_df, vectorizer, tfidf_matrix)
        overall, by_type = summarize_results(all_eval)
        mcnemar_results = run_mcnemar_test(all_eval)

        st.subheader("Overall Top-5 Accuracy")
        st.dataframe(overall, use_container_width=True)

        st.subheader("Top-5 Accuracy by Query Type")
        st.dataframe(by_type, use_container_width=True)
        st.subheader("McNemar Statistical Significance Test")
        st.write("Comparison: TF-IDF vs Expanded Hybrid")
        st.json(mcnemar_results)

        st.subheader("Detailed Evaluation")
        st.dataframe(all_eval, use_container_width=True)

        overall.to_csv("overall_results.csv", index=False, encoding="utf-8-sig")
        by_type.to_csv("results_by_query_type.csv", index=False, encoding="utf-8-sig")
        all_eval.to_csv("detailed_evaluation_results.csv", index=False, encoding="utf-8-sig")

        st.success("Results saved as CSV files in the project folder.")

    st.subheader("Hybrid Weight Tuning Experiment")

    if st.button("Run Weight Tuning"):
        with st.spinner("Running weight tuning experiment... please wait"):
            tuning_df = run_weight_tuning(corpus_df, queries_df)

        st.success("Weight tuning completed.")
        st.dataframe(tuning_df, use_container_width=True)

except FileNotFoundError:
    st.error("CSV files not found. Make sure dari_clean_corpus.csv and dari_evaluation_queries_60.csv are in the same folder as app.py.")
except Exception as e:
    st.error(f"Error: {e}")