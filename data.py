from pymongo import MongoClient

# Connect to MongoDB server
client = MongoClient('localhost', 27017)

# Access a database
db = client['stock_prices']

# Optional: Print available collections in the database
print(db.list_collection_names())
