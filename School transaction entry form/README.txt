Creates a form that connects to PostgreSQL and access the database hist_data
to either insert data, remove a data entry, and show the last 30 rows of data based on date (tanggal). 

1. pengeluaran: the total amount spent, integer
2. komponen dana bos: the categorization of spending (pengeluaran) based on Permendikbudristek Nomor 2 Tahun 2022, string
3. keterangan: Details of the spending amount, string
4. nomor resi: a primary key of PostgreSQL is used to document receipt transactions using a mix of numbers and letters, string
5. tanggal: date of the transaction, datetime

