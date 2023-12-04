import psycopg2
import tkinter as tk
from tkinter import ttk
import sys

username = sys.argv[1] if len(sys.argv) > 1 else ""
password_u = sys.argv[2] if len(sys.argv) > 2 else ""

def connect_db():
    try:
        conn = psycopg2.connect(dbname=db_name, 
                                user=username, 
                                host=host, 
                                port=port, 
                                password=password_u
                                )
        return conn
    except psycopg2.Error as e:
        ttk.messagebox.showwarning(title="Login Failed", message="Username/Password is wrong! Please try again")



# Creating a delete sequence, where if you highlight a row in treeview,
# you can delete it with a button
def on_select(event):
    selected_item = tree.focus()  # Get the selected item
    values = tree.item(selected_item, 'values')

    if values:  # Check if a row is selected
        delete_button.config(state='normal')  # Enable the delete button
    else:
        delete_button.config(state='disabled')  # Disable the delete button


def delete_selected_row():
    selected_item = tree.focus()
    if selected_item:
        # Get the values of the selected row
        values = tree.item(selected_item, 'values')

        # values[3] = nomor_resi which is primary key
        unique_id = values[3]
        # Connect to the database and perform deletion
        try:
            conn = psycopg2.connect(dbname=db_name, user=username, host=host, port=port, password=password_u)
            cursor = conn.cursor()

            # Replace the table if needed and id column
            delete_query = f"DELETE FROM t_schema.pengeluaran_dana_bos WHERE nomor_resi = %s"
            cursor.execute(delete_query, (unique_id,))
            conn.commit()

            cursor.close()
            conn.close()
            # Remove the row from the Treeview
            tree.delete(selected_item)
        except psycopg2.Error as e:
                    print("Unable to delete row:", e)



db_name = 'hist_data'
host = 'localhost'
port='5432'
conn = psycopg2.connect(dbname="hist_data", 
                        user=username, 
                        host=host, 
                        port=port, 
                        password=password_u
                        )

cur = conn.cursor()

# Selects everything from t_schema.pengeluaran dana_bos,
# and shows the last 30 rows based on date
cur.execute("""SELECT * FROM t_schema.pengeluaran_dana_bos ORDER BY tanggal DESC LIMIT 30;""")
headers = [desc[0] for desc in cur.description]
data_rows = cur.fetchall()

cur.close()
conn.close()


# A function to display the collection of tuples collected from postgresql to become a table
root = tk.Tk()
root.title("Table Display")

frame = ttk.Frame(root)
frame.pack(fill='both', expand=True, padx=10, pady=10)

# Create Treeview widget
tree = ttk.Treeview(frame, columns=list(range(len(data_rows[0]))), show="headings")

# Add column headings
for i, heading in enumerate(headers):
    tree.heading(i, text=heading )

# Add data rows
for row in data_rows[0:]:
    tree.insert("", "end", values=row)

# Add a scrollbar
scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scroll_y.set)
scroll_y.pack(side='right', fill='y')

tree.pack(fill='both', expand=True)

tree.bind('<<TreeviewSelect>>', on_select)
global delete_button
delete_button = ttk.Button(root, text="Delete Selected Row", command=delete_selected_row)
delete_button.pack(pady=10)
delete_button.config(state='disabled')  # Initially disable the delete button

root.mainloop()
