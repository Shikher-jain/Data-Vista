import json
import os
import streamlit as st

# File to store data
DATA_FILE = "students.json"

# Load data from file
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Save data to file
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Initialize dictionary
student_grades = load_data()

# Streamlit UI
st.title("🎓 Student Grade Management System")

menu = ["Add Student", "Update Student", "Delete Student", "View Students"]
choice = st.sidebar.selectbox("Menu", menu)

# Add Student
if choice == "Add Student":
    st.subheader("➕ Add New Student")
    name = st.text_input("Student Name")
    grade = st.number_input("Grade (0-100)", min_value=0, max_value=100, step=1)

    if st.button("Add Student"):
        if not name:
            st.warning("Please enter a student name.")
        else:
            student_grades[name] = grade
            save_data(student_grades)
            st.success(f"Added {name} with grade {grade}")

# Update Student
elif choice == "Update Student":
    st.subheader("✏️ Update Student Grade")
    name = st.text_input("Student Name")
    grade = st.number_input("New Grade (0-100)", min_value=0, max_value=100, step=1)

    if st.button("Update"):
        if name in student_grades:
            student_grades[name] = grade
            save_data(student_grades)
            st.success(f"Updated {name}'s grade to {grade}")
        else:
            st.error("Student not found")

# Delete Student
elif choice == "Delete Student":
    st.subheader("🗑️ Delete Student")
    name = st.text_input("Student Name to Delete")

    if st.button("Delete"):
        if name in student_grades:
            del student_grades[name]
            save_data(student_grades)
            st.success(f"Deleted {name}")
        else:
            st.error("Student not found")

# View Students
elif choice == "View Students":
    st.subheader("📋 Student List")

    if student_grades:
        st.table([{"Name": name, "Grade": grade} for name, grade in student_grades.items()])
    else:
        st.info("No students found")
