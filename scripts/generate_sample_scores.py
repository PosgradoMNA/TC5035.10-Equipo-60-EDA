import json
import time
import requests
import pandas as pd
from pathlib import Path

JOB_TITLES = [
    'Digital Marketing Specialist', 'Web Developer', 'Operations Manager', 'Network Engineer',
    'Event Manager', 'Software Tester', 'Teacher', 'UX/UI Designer', 'Wedding Planner',
    'QA Analyst', 'Litigation Attorney', 'Mechanical Engineer', 'Network Administrator',
    'Account Manager', 'Brand Manager', 'Social Worker', 'Social Media Coordinator',
    'Email Marketing Specialist', 'HR Generalist', 'Legal Assistant', 'Nurse Practitioner',
    'Account Director', 'Software Engineer', 'Purchasing Agent', 'Sales Consultant',
    'Civil Engineer', 'Network Security Specialist', 'UI Developer', 'Financial Planner',
    'Event Planner', 'Psychologist', 'Electrical Designer', 'Data Analyst', 'Technical Writer',
    'Tax Consultant', 'Account Executive', 'Systems Administrator', 'Database Administrator',
    'Research Analyst', 'Data Entry Clerk', 'Registered Nurse', 'Investment Analyst',
    'Speech Therapist', 'Sales Manager', 'Landscape Architect', 'Key Account Manager',
    'UX Researcher', 'Investment Banker', 'IT Support Specialist', 'Art Director',
    'Software Developer', 'Project Manager', 'Customer Service Manager', 'Procurement Manager',
    'Substance Abuse Counselor', 'Supply Chain Analyst', 'Data Engineer', 'Accountant',
    'Sales Representative', 'Environmental Consultant', 'Electrical Engineer', 'Systems Engineer',
    'Art Teacher', 'Human Resources Manager', 'Inventory Analyst', 'Legal Counsel',
    'Database Developer', 'Procurement Specialist', 'Systems Analyst', 'Copywriter',
    'Content Writer', 'HR Coordinator', 'Business Development Manager', 'Java Developer',
    'Supply Chain Manager', 'Event Coordinator', 'Family Nurse Practitioner', 'Front-End Engineer',
    'Customer Success Manager', 'Procurement Coordinator', 'Urban Planner',
    'Architectural Designer', 'Financial Analyst', 'Environmental Engineer', 'Back-End Developer',
    'Structural Engineer', 'Market Research Analyst', 'Customer Service Representative',
    'Customer Support Specialist', 'Business Analyst', 'Social Media Manager', 'Family Lawyer',
    'Chemical Analyst', 'Network Technician', 'Interior Designer', 'Software Architect',
    'Nurse Manager', 'Veterinarian', 'Process Engineer', 'IT Manager',
    'Quality Assurance Analyst', 'Pharmaceutical Sales Representative', 'Office Manager',
    'Architect', 'Physician Assistant', 'Marketing Director', 'Front-End Developer',
    'Research Scientist', 'Executive Assistant', 'HR Manager', 'Marketing Manager',
    'Public Relations Specialist', 'Financial Controller', 'Investment Advisor',
    'Aerospace Engineer', 'Marketing Analyst', 'Paralegal', 'Landscape Designer',
    'Web Designer', 'Occupational Therapist', 'Legal Advisor', 'Marketing Coordinator',
    'Dental Hygienist', 'SEM Specialist', 'SEO Specialist', 'Pediatrician', 'QA Engineer',
    'Data Scientist', 'Financial Advisor', 'Personal Assistant', 'SEO Analyst',
    'Network Analyst', 'Mechanical Designer', 'Marketing Specialist', 'Graphic Designer',
    'Finance Manager', 'Physical Therapist', 'Product Designer', 'Administrative Assistant',
    'Brand Ambassador', 'Project Coordinator', 'Product Manager', 'IT Administrator',
    'Sales Associate', 'Chemical Engineer', 'Legal Secretary', 'Market Analyst'
]

DATASET_PATH = Path("/Users/cesarivp/Documents/GitHub/TC5035.10-Equipo-60-EDA/data/output/featured_engineered_dataset.csv")
OUTPUT_DIR = Path("/Users/cesarivp/Documents/GitHub/TC5035.10-Equipo-60-EDA/data/output/samples")
ROWS_PER_COMPARATOR = 200
OFFSET = 1  # 0 = original run, 1+ = append new rows
PROGRESS_FILE = OUTPUT_DIR / f"_progress_offset{OFFSET}.json"
OLLAMA_URL = "http://localhost:11434/api/generate"


def get_score(comparator: str, row: pd.Series) -> float:
    """Score a row against a comparator using Ollama."""
    if row["Job Title"] == comparator:
        return 5.0

    prompt = (
        f"Rate how aligned this job profile is to the role '{comparator}' on a scale from 0.0 to 5.0.\n"
        f"5.0 = perfect match, 0.0 = completely unrelated.\n\n"
        f"Job Title: {row['Job Title']}\n"
        f"Role: {row['Role']}\n"
        f"Qualifications: {row['Qualifications']}\n"
        f"Experience: {row['Experience_avg']} years\n"
        f"Skills: {row['skills'][:200]}\n\n"
        f"Consider skills overlap, experience relevance, and career path similarity.\n"
        f"Respond with ONLY a single number (e.g., 3.5). Nothing else."
    )

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=60)
        text = resp.json()["response"].strip()
        for token in text.replace(",", ".").split():
            try:
                score = float(token)
                return max(0.0, min(5.0, score))
            except ValueError:
                continue
        return 2.5
    except Exception as e:
        print(f"  Error: {e}")
        return 2.5


def main():
    print(f"Config: rows={ROWS_PER_COMPARATOR}, offset={OFFSET}")
    print("Loading dataset (this may take a minute)...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} rows")

    # Load progress
    done_comparators = set()
    if PROGRESS_FILE.exists():
        done_comparators = set(json.loads(PROGRESS_FILE.read_text()))
        print(f"Resuming — {len(done_comparators)}/147 comparators already done for offset {OFFSET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for i, comparator in enumerate(JOB_TITLES):
        if comparator in done_comparators:
            continue

        print(f"\n[{i+1}/147] Comparator: {comparator}")

        # Use OFFSET to get different samples each run
        sample = df.sample(n=ROWS_PER_COMPARATOR, random_state=i + (OFFSET * 1000))

        scores = []
        for idx, (_, row) in enumerate(sample.iterrows()):
            score = get_score(comparator, row)
            scores.append(score)
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  {idx+1}/{ROWS_PER_COMPARATOR} scored | {elapsed/60:.1f} min elapsed")

        sample = sample.copy()
        sample["Job_Comparator"] = comparator
        sample["Score"] = scores

        # Save or append
        filename = comparator.lower().replace("/", "-").replace(" ", "_") + "_scores.csv"
        filepath = OUTPUT_DIR / filename

        if OFFSET == 0:
            sample.to_csv(filepath, index=False)
        else:
            sample.to_csv(filepath, mode='a', header=False, index=False)

        # Update progress
        done_comparators.add(comparator)
        PROGRESS_FILE.write_text(json.dumps(list(done_comparators)))
        print(f"  Saved: {filename} (avg score: {sum(scores)/len(scores):.2f})")

    print(f"\nDone! {len(done_comparators)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
