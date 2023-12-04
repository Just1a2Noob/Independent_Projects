import psycopg2
import tkinter as tk
from tkinter import messagebox
import subprocess

def open_main(username, password):
    file_to_run = r'C:\Users\ThinkPad\Documents\Anaconda_RootDIR\Projects\tkinter_form\main.py'
    command = ['python', file_to_run, username, password]
    window.destroy()
    subprocess.run(command, check=True)

def check_access():
    username = username_entry.get()
    password = password_entry.get()

    db_name="hist_data"
    host = "localhost"
    port= '5432'

    try:
        conn = psycopg2.connect(dbname=db_name, 
                                user=username, 
                                host=host, 
                                port=port, 
                                password=password
                                )
        return conn, open_main(username, password)
    except psycopg2.Error as e:
        tk.messagebox.showwarning(title="Login Failed", message="Username/Password is wrong! Please try again")

window = tk.Tk()
window.title("Account Login")

frame = tk.Frame(window)
frame.pack(padx=20, pady=20)

username_label = tk.Label(frame, text="Username:")
username_label.grid(row=0, column=0, padx=5, pady=5)
username_entry = tk.Entry(frame)
username_entry.grid(row=0, column=1, padx=5, pady=5)

password_label = tk.Label(frame, text="Password:")
password_label.grid(row=1, column=0, padx=5, pady=5)
password_entry = tk.Entry(frame, show="*")
password_entry.grid(row=1, column=1, padx=5, pady=5)

submit_button = tk.Button(frame, text="Submit", command=check_access)
submit_button.grid(row=2, columnspan=2, padx=5, pady=5)

result_label = tk.Label(frame, text="")
result_label.grid(row=3, columnspan=2, padx=5, pady=5)

window.mainloop()


