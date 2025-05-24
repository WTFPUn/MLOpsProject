import pandas as pd
import argparse
from itertools import combinations
import random
import os

EXCLUDE_TAGS = {
    'ข่าววันนี้', 'ข่าวทั่วไทย', 'ข่าวการเมืองวันนี้', 'ข่าวการเมือง', 'ข่าวด่วน',
    'ข่าวการเมือง ไทยรัฐ', 'กรมอุตุนิยมวิทยา', 'เตือนฝนตก', 'ไทยรัฐฉบับพิมพ์',
    'ข่าวหน้า1', 'สภาพอากาศวันนี้', 'ประกาศกรมอุตุ', 'พยากรณ์อากาศ'
}

def tags_to_set(row):
    if isinstance(row, str):
        return set(
            t.strip() for t in row.split(",")
            if t.strip() and t.strip() not in EXCLUDE_TAGS
        )
    return set()

def main(args):
    news = pd.read_csv(args.input_csv)
    news["tag_set"] = news["tags"].apply(tags_to_set)

    min_overlap_count = 4
    min_overlap_ratio = 0.4

    positive_pairs = []
    for i, j in combinations(range(len(news)), 2):
        tags_i, tags_j = news.at[i, "tag_set"], news.at[j, "tag_set"]
        overlap = tags_i & tags_j
        if len(overlap) >= min_overlap_count:
            ratio_i = len(overlap) / len(tags_i) if tags_i else 0
            ratio_j = len(overlap) / len(tags_j) if tags_j else 0
            if ratio_i >= min_overlap_ratio or ratio_j >= min_overlap_ratio:
                positive_pairs.append((i, j))

    print(f"Found {len(positive_pairs)} positive pairs")

    triplets = []
    all_indices = set(range(len(news)))

    for i, j in positive_pairs:
        tags_i = news.at[i, "tag_set"]
        neg_candidates = [k for k in all_indices - {i, j} if news.at[k, "tag_set"].isdisjoint(tags_i)]
        sampled_neg = random.sample(neg_candidates, min(len(neg_candidates), args.negatives_per_pair))

        for k in sampled_neg:
            triplets.append({
                "anchor": news.at[i, "content"],
                "positive": news.at[j, "content"],
                "negative": news.at[k, "content"]
            })

    df_triplet = pd.DataFrame(triplets)
    print(f"Generated {len(df_triplet)} triplets")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_triplet.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate triplet data from Thai news tags")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input news CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output triplet CSV")
    parser.add_argument("--negatives_per_pair", type=int, default=5, help="Number of negatives per anchor-positive pair")

    args = parser.parse_args()
    main(args)
