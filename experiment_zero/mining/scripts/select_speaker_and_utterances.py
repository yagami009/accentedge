#!/usr/bin/env python3
"""Select one Vaani speaker + 30 diagnostic utterances for Experiment Zero."""
import pandas as pd
import re
from pathlib import Path

BANG_EN = Path("/Volumes/AYUSH_SSD/accentedge-data/processed/manifests/utterances_vaani_bangalore_en.parquet")
HYD_EN = Path("/Volumes/AYUSH_SSD/accentedge-data/processed/manifests/utterances_vaani_hyderabad.parquet")
OUT_DIR = Path("/Users/ayushmh/accentedge/experiment_zero/mining/candidates")

# Diagnostic categories from frozen matrix
DIAGNOSTIC_KEYWORDS = {
    "rhoticity": ["order", "processed", "ship", "tomorrow", "correct", "three"],
    "th_sounds": ["three", "thirteen", "thirty", "thursday"],
    "vw_confusion": ["seven", "very", "have", "weeks"],
    "aspiration": ["technical", "support", "transfer", "warranty", "three", "years"],
    "vowel_reduction": ["about", "problem", "support", "account", "system"],
    "consonant_clusters": ["technical", "support", "transfer", "warranty", "strengths"],
    "lexical_stress": ["discount", "percent", "reservation", "restaurant"],
    "critical_pairs": ["fifteen", "fifty", "thirteen", "thirty", "fourteen", "forty"],
    "hinglish": ["rahul", "mumbai", "aap", "account", "main"],
    "connected_speech": ["actually", "anything", "assist", "moment", "pull"],
    "proper_nouns": ["rahul", "mumbai", "johnson", "acme"],
    "numbers_dates": ["fifteen", "fifty", "thirteen", "thirty", "thursday", "march"],
    "acronyms": ["bpo"],
    "negation": ["can't", "cannot", "didn't", "not"],
    "dollar_amounts": ["dollars", "cents", "payment"],
    "long_numeric": ["confirmation", "number", "seven", "four", "three"],
}

def score_utterance(text, diag_tags):
    """Score how well an utterance covers the diagnostic matrix."""
    text_lower = text.lower()
    score = 0
    matched = []
    for tag, keywords in diag_tags.items():
        if any(kw in text_lower for kw in keywords):
            score += 1
            matched.append(tag)
    return score, matched

def select_diagnostic_utterances(df, n=30):
    """Select n utterances maximizing diagnostic coverage."""
    df = df.copy()
    df['text_clean'] = df['transcript'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)).strip() if pd.notna(x) else '')
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
    
    # Filter: 3-15 words, clean text
    df = df[(df['word_count'] >= 3) & (df['word_count'] <= 15) & (df['text_clean'].str.len() > 5)]
    
    # Score each utterance
    scores = []
    for idx, row in df.iterrows():
        score, matched = score_utterance(row['text_clean'], DIAGNOSTIC_KEYWORDS)
        scores.append((idx, score, matched))
    
    df['diag_score'] = [s[1] for s in scores]
    df['diag_matched'] = [s[2] for s in scores]
    
    # Greedy selection: pick highest-scoring, then fill gaps
    selected = []
    covered_tags = set()
    
    # First pass: ensure all diagnostic categories are covered
    for tag in DIAGNOSTIC_KEYWORDS.keys():
        candidates = df[df['diag_matched'].apply(lambda x: tag in x)]
        if len(candidates) > 0:
            best = candidates.loc[candidates['diag_score'].idxmax()]
            selected.append(best)
            covered_tags.update(best['diag_matched'])
            df = df.drop(best.name)
    
    # Second pass: fill remaining slots with highest-scoring
    remaining = n - len(selected)
    if remaining > 0:
        filler = df.nlargest(remaining, 'diag_score')
        selected.extend([filler.iloc[i] for i in range(min(remaining, len(filler)))])
    
    return pd.DataFrame(selected)

if __name__ == "__main__":
    # Wait for manifests to exist
    if not BANG_EN.exists():
        print("Waiting for Vaani Bangalore English manifest...")
        exit(0)
    
    print("Loading Vaani English manifests...")
    bang = pd.read_parquet(BANG_EN)
    hyd = pd.read_parquet(HYD_EN)
    
    print(f"Bangalore English: {len(bang)} utterances")
    print(f"Hyderabad English: {len(hyd)} utterances")
    
    # Select speaker with most utterances
    print("\n=== Top Speakers (Bangalore) ===")
    bang_spk = bang.groupby('speakerID').agg({
        'UtteranceSequenceID': 'count',
        'duration': ['sum', 'mean', 'min', 'max']
    }).sort_values(('UtteranceSequenceID', 'count'), ascending=False)
    bang_spk.columns = ['count', 'total_dur', 'avg_dur', 'min_dur', 'max_dur']
    print(bang_spk.head(10))
    
    # Select from best speaker
    best_spk = bang_spk.index[0]
    print(f"\nSelected speaker: {best_spk} ({bang_spk.loc[best_spk, 'count']} utterances)")
    
    spk_utts = bang[bang['speakerID'] == best_spk]
    selected = select_diagnostic_utterances(spk_utts, n=30)
    
    print(f"\nSelected {len(selected)} utterances")
    print("\nDiagnostic coverage:")
    all_tags = []
    for tags in selected['diag_matched']:
        all_tags.extend(tags)
    from collections import Counter
    tag_counts = Counter(all_tags)
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")
    
    # Save selection
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(OUT_DIR / 'selected_vaani_utterances.parquet', index=False)
    print(f"\nSaved: {OUT_DIR / 'selected_vaani_utterances.parquet'}")
