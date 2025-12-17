from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoManager:
    def __init__(
        self,
        collection_name: str = "admin_user",
        db_name: str = "portfolio_data",
        mongo_uri: str = None,
    ):
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGO_URI", "mongodb://localhost:27017"
        )
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.connect()

    def connect(self):
        """Establish connection to MongoDB with timeout validation."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000
            )
            self.client.server_info()  # Force connection check
            self.collection = self.client[self.db_name][self.collection_name]
            logger.info(
                f"Connected to MongoDB | DB: {self.db_name} | Collection: {self.collection_name}"
            )
        except PyMongoError as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise RuntimeError("MongoDB connection error") from e

    # ---------------- CRUD OPERATIONS ---------------- #

    def insert_document(self, document: dict):
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Insert failed: {e}")
            return None

    def insert_documents(self, documents: list):
        try:
            result = self.collection.insert_many(documents)
            return [str(_id) for _id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"Bulk insert failed: {e}")
            return None

    def find_document(self, query: dict):
        try:
            if "_id" in query and isinstance(query["_id"], str):
                query["_id"] = ObjectId(query["_id"])
            return self.collection.find_one(query)
        except PyMongoError as e:
            logger.error(f"Find one failed: {e}")
            return None

    def find_documents(self, query: dict):
        try:
            return list(self.collection.find(query))
        except PyMongoError as e:
            logger.error(f"Find many failed: {e}")
            return None

    def update_document(self, query: dict, update_fields: dict):
        try:
            if "_id" in query and isinstance(query["_id"], str):
                query["_id"] = ObjectId(query["_id"])

            update = {"$set": update_fields}
            result = self.collection.update_one(query, update)
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Update failed: {e}")
            return None



    def delete_document(self, query: dict):
        try:
            if "_id" in query and isinstance(query["_id"], str):
                query["_id"] = ObjectId(query["_id"])
            result = self.collection.delete_one(query)
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Delete failed: {e}")
            return None

    def close_connection(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
