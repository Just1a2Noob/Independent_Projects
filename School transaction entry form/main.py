import subprocess
import sys
import tkinter
from datetime import datetime
from tkinter import messagebox, ttk

import psycopg

# Getting the username and password from connectio.py
username = sys.argv[1] if len(sys.argv) > 1 else ""
password_u = sys.argv[2] if len(sys.argv) > 2 else ""
db_name = "hist_data"
host = "localhost"
port = "5432"


def open_table():
    # Replace the value of file_to_run with the name of the Python file you want to run
    file_to_run = r"C:\Users\ThinkPad\Documents\Anaconda_RootDIR\Projects\tkinter_form\table_show.py"
    command = ["python", file_to_run, username, password_u]
    subprocess.run(command, check=True)


def clear_entries_and_comboboxes():
    for frame in all_frames:
        # Iterate through the widgets in the LabelFrame
        for widget in frame.winfo_children():
            if isinstance(widget, tkinter.Entry):
                widget.delete(0, tkinter.END)  # Clear Entry widget
            elif isinstance(widget, ttk.Combobox):
                widget.set("")  # Reset Combobox selection


# Form for entering data
def enter_data():
    conn = psycopg2.connect(
        dbname=db_name, user=username, host=host, port=port, password=password_u
    )
    cur = conn.cursor()
    pengeluaran = pengeluaran_entry.get()
    komponen_dana_bos = komponen_dana_bos_combobox.get()
    keterangan = keterangan_entry.get()
    nomor_resi = nomor_resi_entry.get()
    tanggal = tanggal_entry.get()
    if check_date_format(tanggal) == True:
        insert_query = f"INSERT INTO t_schema.pengeluaran_dana_bos (pengeluaran, komponen_dana_bos, keterangan, nomor_resi, tanggal) VALUES (%s, %s, %s, %s, %s)"
        row_input = (pengeluaran, komponen_dana_bos, keterangan, nomor_resi, tanggal)
        cur.execute(insert_query, row_input)
        conn.commit()
        cur.close()
        conn.close()
        return clear_entries_and_comboboxes()
    else:
        tkinter.messagebox.showwarning(
            title="Error",
            message="Format tanggal harus dengan bentuk Tahun-Bulan-Hari, contoh 2014-08-17",
        )


def check_date_format(date_str):
    try:
        # Attempt to parse the input string into a datetime object
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# clear_example and show_example are for date format
def clear_example(event):
    # Function to clear the example text when Entry is clicked
    if tanggal_entry.get() == "TAHUN-BULAN-HARI":
        tanggal_entry.delete(0, tkinter.END)


def show_example(event):
    # Function to show the example text if Entry is left empty
    if not tanggal_entry.get():
        tanggal_entry.insert(0, "TAHUN-BULAN-HARI")


def show_description(event):
    # Function to display description based on selected option
    selected_option = (
        komponen_dana_bos_combobox.get()
    )  # Get the selected option from the Combobox

    # Define descriptions associated with each option
    # This is from https://ditsmp.kemdikbud.go.id/daftar-terbaru-komponen-penggunaan-bos-reguler/
    descriptions = {
        "Penerimaan Peserta Didik baru": "Contoh kegiatan dalam rangka penerimaan peserta didik baru antara lain seperti penggandaan formulir pendaftaran, penerimaan Peserta Didik baru dalam jaringan, publikasi atau pengumuman penerimaan Peserta Didik baru, kegiatan pengenalan lingkungan Satuan Pendidikan untuk anak dan orang tua, pendataan ulang Peserta Didik lama, dan/atau kegiatan lain yang relevan dalam rangka pelaksanaan penerimaan Peserta Didik baru.",
        "Pengembangan perpustakaan": "Komponen pengembangan perpustakaan yang dapat dibiayai menggunakan dana BOS reguler antara lain seperti penyediaan buku teks utama termasuk buku digital, penyediaan buku teks pendamping, penyediaan buku nonteks termasuk buku digital, penyediaan atau pencetakan modul dan perangkat ajar; dan/ atau pembiayaan lain yang relevan dalam rangka pengembangan perpustakaan.",
        "Pelaksanaan kegiatan pembelajaran dan ekstrakurikuler": "Dalam pelaksanaan kegiatan pembelajaran, beberapa komponen yang dapat dibiayai dari dana BOS Reguler antara lain penyediaan alat pendidikan dan bahan pendukung pembelajaran, biaya untuk mengembangkan media pembelajaran berbasis teknologi informasi dan komunikasi, penyediaan aplikasi atau perangkat lunak untuk pembelajaran, dan kegiatan pembelajaran lain yang relevan dalam rangka menunjang proses pembelajaran. Sedangkan dalam pelaksanaan ekstrakurikuler, komponen yang dapat dibiayai antara lain seperti penyelenggaraan ekstrakurikuler yang sesuai dengan kebutuhan sekolah, pembiayaan dalam rangka mengikuti lomba, dan atau pembiayaan lain yang relevan dalam rangka menunjang operasional kegiatan ekstrakurikuler.",
        "Pelaksanaan kegiatan asesmen dan evaluasi pembelajaran": "Pelaksanaan kegiatan asesmen dan evaluasi pembelajaran yang dimaksudkan antara lain seperti penyelenggaraan ulangan harian, ulangan tengah semester, ulangan akhir semester, ulangan kenaikan kelas, asesmen nasional, survei karakter, asesmen sekolah, asesmen berbasis komputer dan/atau asesmen lainnya dan atau pembiayaan lain yang relevan untuk kegiatan asesmen dan evaluasi pembelajaran di sekolah.",
        "Pelaksanaan administrasi kegiatan sekolah": "Adapun contoh komponen pelaksanaan administrasi kegiatan sekolah yang dapat dibiayai dari dana BOS Reguler seperti pengelolaan dan operasional rutin sekolah baik dalam rangka pembelajaran tatap muka dan/atau pembelajaran jarak jauh, pembelian sabun pembersih tangan, cairan disinfektan, masker dan penunjang lainnya, dan atau pembiayaan lainnya yang relevan dalam rangka pemenuhan administrasi kegiatan sekolah.",
        "Pengembangan profesi guru dan tenaga kependidikan": "Kegiatan yang dimaksud dalam rangka pengembangan profesi guru dan tenaga kependidikan antara lain pengembangan/peningkatan kompetensi guru dan tenaga kependidikan, pengembangan inovasi terkait konten pembelajaran dan metode pembelajaran, dan atau pembiayaan lain yang relevan dalam rangka menunjang pengembangan profesi guru dan tenaga kependidikan.",
        "Pembiayaan langganan daya dan jasa": "Pembiayaan yang dimaksud antara lain seperti pembiayaan listrik, internet, dan air, penyediaan obat-obatan, peralatan kebersihan atau peralatan kesehatan lainnya dalam rangka menjaga kesehatan Peserta Didik dan pendidik, dan atau pembiayaan lain yang relevan dalam rangka pemenuhan kebutuhan daya dan atau jasa Satuan Pendidikan.",
        "Pemeliharaan sarana dan prasarana sekolah": "Dalam hal pemeliharaan sarana dan prasarana sekolah kegiatan seperti pemeliharaan alat pembelajaran, pemeliharaan alat peraga pendidikan, dan atau pembiayaan lain yang relevan dalam rangka pemeliharaan sarana dan prasarana Satuan Pendidikan juga dapat menggunakan dana BOS Reguler.",
        "Penyediaan alat multimedia pembelajaran": "Penyediaan alat multimedia pembelajaran seperti pencetakan atau pengadaan modul, penyusunan modul interaktif dan media pembelajaran berbasis teknologi informasi dan komunikasi, pengadaan alat keterampilan, bahan praktik keterampilan, komputer desktop dan/atau laptop untuk digunakan dalam proses pembelajaran; dan atau alat multimedia pembelajaran lainnya yang relevan dalam rangka menunjang pembelajaran berbasis teknologi informasi dan komunikasi. ",
        "Penyelenggaraan kegiatan peningkatan kompetensi keahlian": "Kegiatan yang relevan dalam rangka meningkatkan kompetensi keahlian menjadi salah satu komponen yang dapat menggunakan dana BOS Reguler yang diterima satuan pendidikan.",
        "Penyelenggaraan kegiatan dalam mendukung keterserapan lulusan": "Kegiatan yang relevan dalam rangka mendukung keterserapan lulusan menjadi salah satu komponen yang dapat menggunakan dana BOS Reguler yang diterima satuan pendidikan",
        "Pembayaran honor": "Pembayaran honor dapat digunakan paling banyak 50% dari keseluruhan jumlah alokasi Dana BOS Reguler yang diterima oleh Satuan Pendidikan. Pembayaran honor dapat diberikan kepada guru berstatus bukan aparatur sipil negara, tercatat pada Dapodik, memiliki nomor unik pendidik dan tenaga kependidikan, dan belum mendapatkan tunjangan profesi guru.",
    }

    # Display the description in the label
    description_label.config(
        text=descriptions.get(selected_option, "No description available")
    )


window = tkinter.Tk()
window.title("Data Entry Form")

frame = tkinter.LabelFrame(window)
frame.pack()

# Username beserta pengeluaran dan nomor_resi
user_info_frame = tkinter.LabelFrame(frame, text=username)
user_info_frame.grid(row=0, column=0, padx=20, pady=10)

pengeluaran_label = tkinter.Label(user_info_frame, text="Jumlah Pengeluaran")
pengeluaran_label.grid(row=0, column=0)
nomor_resi_label = tkinter.Label(user_info_frame, text="Nomor Resi")
nomor_resi_label.grid(row=0, column=1)

## Creates a entry box for the labels within the frame
pengeluaran_entry = tkinter.Entry(user_info_frame)
nomor_resi_entry = tkinter.Entry(user_info_frame)
pengeluaran_entry.grid(row=1, column=0)
nomor_resi_entry.grid(row=1, column=1)

# Creates an entry for date
tanggal_label = tkinter.Label(user_info_frame, text="Tanggal", fg="black")
tanggal_entry = tkinter.Entry(user_info_frame)
tanggal_label.grid(row=0, column=2)

tanggal_entry.insert(0, "TAHUN-BULAN-HARI")
tanggal_entry.bind("<FocusIn>", clear_example)
tanggal_entry.bind("<FocusOut>", show_example)
tanggal_entry.grid(row=1, column=2)

## This creates padding (spacing between labels) for each label in user_info_frame
## this method is a shortcut to adding padx and pady to each grid
for widget in user_info_frame.winfo_children():
    widget.grid_configure(padx=10, pady=5)

for child in user_info_frame.winfo_children():
    if isinstance(child, tkinter.Entry):
        child.config(width=(25))


komponen_dana_bos_frame = tkinter.LabelFrame(frame)
komponen_dana_bos_frame.grid(row=1, column=0, padx=20, pady=10)

komponen_dana_bos_label = tkinter.Label(
    komponen_dana_bos_frame, text="Komponen Dana Boss"
)
komponen_dana_bos_label.grid(row=0, column=0)

komponens = [
    "Penerimaan Peserta Didik baru",
    "Pengembangan perpustakaan",
    "Pelaksanaan kegiatan pembelajaran dan ekstrakurikuler",
    "Pelaksanaan kegiatan asesmen dan evaluasi pembelajaran",
    "Pelaksanaan administrasi kegiatan sekolah",
    "Pengembangan profesi guru dan tenaga kependidikan",
    "Pembiayaan langganan daya dan jasa",
    "Pemeliharaan sarana dan prasarana sekolah",
    "Penyediaan alat multimedia pembelajaran",
    "Penyelenggaraan kegiatan peningkatan kompetensi keahlian",
    "Penyelenggaraan kegiatan dalam mendukung keterserapan lulusan",
    "Pembayaran honor",
]
selected_option = tkinter.StringVar()
komponen_dana_bos_combobox = ttk.Combobox(
    komponen_dana_bos_frame, textvariable=selected_option, values=komponens, width=60
)
komponen_dana_bos_combobox.grid(row=1, column=0)


description_label = tkinter.Label(
    komponen_dana_bos_frame,
    text="Select an option to see its description",
    wraplength=400,
)
description_label.grid(row=2, column=0, sticky="news")
# Bind the function to the Combobox selection event
komponen_dana_bos_combobox.bind("<<ComboboxSelected>>", show_description)

for widget in komponen_dana_bos_frame.winfo_children():
    widget.grid_configure(padx=10, pady=5)


keterangan_frame = tkinter.LabelFrame(frame)
keterangan_frame.grid(row=3, column=0, padx=20, pady=10)

keterangan_label = tkinter.Label(keterangan_frame, text="Keterangan")
keterangan_entry = tkinter.Entry(keterangan_frame, width=60)
keterangan_label.grid(row=0, column=0, padx=10, pady=5)
keterangan_entry.grid(row=1, column=0, padx=20, pady=5)

# Button
button_enter = tkinter.Button(frame, text="Enter Data", command=enter_data)
button_enter.grid(row=4, column=0, sticky="news", padx=20, pady=10)

button_table = tkinter.Button(frame, text="Show Table", command=open_table)
button_table.grid(row=5, column=0, sticky="news", padx=20)

all_frames = [frame, user_info_frame, keterangan_frame, komponen_dana_bos_frame]

window.mainloop()

