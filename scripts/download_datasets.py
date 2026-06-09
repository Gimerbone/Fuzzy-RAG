"""
Pre-download HuggingFace datasets (PubMedQA and MedQA) to the local cache.
RadQA requires PhysioNet credentialed access — see CLAUDE.md for instructions.

Usage:
    python scripts/download_datasets.py
"""

from datasets import load_dataset


def main():
    print("Downloading PubMedQA (pqa_labeled)...")
    load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
    print("  Done.")

    print("Downloading MedQA (med_qa_en_bigbio_qa)...")
    load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa", trust_remote_code=True)
    print("  Done.")

    print("All public datasets downloaded.")
    print("\nFor RadQA:")
    print("  1. Request access at https://physionet.org/content/radqa/1.0.0/")
    print("  2. Download radqa_train.json (and optionally dev/test) to your RADQA_DATA_PATH directory.")


if __name__ == "__main__":
    main()
