# generate_synthetic_data.py
import json
import random
import pandas as pd

def load_course_requirements(filepath='data/course_requirements.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_subjects(compulsory, electives, total=5):
    if electives == "N/A":
        electives_list = []
    else:
        electives_list = [s.strip() for s in electives.split(",")]
    compulsory_list = [s.strip() for s in compulsory.split(",")]
    remaining = max(0, total - len(compulsory_list))
    selected_electives = random.sample(electives_list, min(remaining, len(electives_list)))
    return compulsory_list + selected_electives

def generate_grades(n):
    return random.choices(["A1", "B2", "B3", "C4", "C5", "C6", "D7", "E8", "F9"], k=n)

def generate_jamb_subjects(requirement):
    subjects = [s.strip() for s in requirement.split(",")]
    sample_size = min(4, len(subjects))  # Don't exceed available subject count
    return random.sample(subjects, sample_size)

def generate_student_record(program_data):
    waec_subjects = generate_subjects(program_data["OLEVEL_Compulsory"], program_data["OLEVEL_Elective"])
    waec_grades = generate_grades(len(waec_subjects))
    jamb_subjects = generate_jamb_subjects(program_data["UTME_Requirement"])
    jamb_score = random.randint(150, 300)
    post_utme_score = random.randint(30, 100)
    aggregate_score = round((jamb_score / 400) * 50 + (post_utme_score / 100) * 50, 2)
    course_cutoff = float(program_data["Aggregate_Score"])
    admitted_course = program_data["Program"] if aggregate_score >= course_cutoff else "NOT ADMITTED"

    return {
        "WAEC_Subjects": ', '.join(waec_subjects),
        "WAEC_Grades": ', '.join(waec_grades),
        "JAMB_Subjects": ', '.join(jamb_subjects),
        "JAMB_Score": jamb_score,
        "JAMB_Cutoff": int(program_data["JAMB_Cutoff"]),
        "Post_UTME_Score": post_utme_score,
        "Aggregate_Score": aggregate_score,
        "Course_Cutoff": course_cutoff,
        "Admitted_Course": admitted_course
    }

def main():
    programs = load_course_requirements()
    all_students = []

    for _ in range(100_000):
        program_data = random.choice(programs)
        record = generate_student_record(program_data)
        record["Program"] = program_data["Program"]
        all_students.append(record)

    df = pd.DataFrame(all_students)
    df.to_csv("synthetic_dataset_100k.csv", index=False)
    print("✅ Generated synthetic_dataset_100k.csv with 100,000 entries")

if __name__ == "__main__":
    main()
