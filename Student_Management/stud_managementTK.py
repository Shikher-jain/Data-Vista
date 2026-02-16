import json
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

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

# --- UI Functions ---
def add_student():
    name = name_entry.get().strip()
    grade = grade_entry.get().strip()

    if not name or not grade:
        messagebox.showwarning("Input Error", "Please enter both name and grade.")
        return

    if not grade.isdigit() or not (0 <= int(grade) <= 100):
        messagebox.showwarning("Input Error", "Grade must be a number between 0 and 100.")
        return

    student_grades[name] = int(grade)
    save_data(student_grades)
    update_list()
    messagebox.showinfo("Success", f"{name} added successfully!")

def update_student():
    name = name_entry.get().strip()
    grade = grade_entry.get().strip()

    if name not in student_grades:
        messagebox.showerror("Error", "Student not found.")
        return

    if not grade.isdigit() or not (0 <= int(grade) <= 100):
        messagebox.showwarning("Input Error", "Grade must be a number between 0 and 100.")
        return

    student_grades[name] = int(grade)
    save_data(student_grades)
    update_list()
    messagebox.showinfo("Updated", f"{name}'s grade updated!")

def delete_student():
    name = name_entry.get().strip()

    if name not in student_grades:
        messagebox.showerror("Error", "Student not found.")
        return

    del student_grades[name]
    save_data(student_grades)
    update_list()
    messagebox.showinfo("Deleted", f"{name} removed.")

def update_list():
    listbox.delete(0, tk.END)
    if not student_grades:
        listbox.insert(tk.END, "No students found")
        return
    for name, grade in student_grades.items():
        listbox.insert(tk.END, f"{name}: {grade}")

# --- UI Setup ---
root = tk.Tk()
root.title("Student Grade Management System")
root.geometry("400x450")
root.config(bg="#F0F0F0")

# Labels and entries
tk.Label(root, text="Student Name:", font=("Arial", 12)).pack(pady=5)
name_entry = tk.Entry(root, width=30)
name_entry.pack()

tk.Label(root, text="Grade (0-100):", font=("Arial", 12)).pack(pady=5)
grade_entry = tk.Entry(root, width=30)
grade_entry.pack()

# Buttons
tk.Button(root, text="Add Student", width=20, command=add_student).pack(pady=10)
tk.Button(root, text="Update Student", width=20, command=update_student).pack(pady=5)
tk.Button(root, text="Delete Student", width=20, command=delete_student).pack(pady=5)

# Listbox to show students
tk.Label(root, text="Student List:", font=("Arial", 12, "bold")).pack(pady=10)
listbox = tk.Listbox(root, width=40, height=10)
listbox.pack()

update_list()

root.mainloop()
