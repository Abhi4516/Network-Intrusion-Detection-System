import os
import sys
import json
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Path to certificate for SSL connection to MongoDB Atlas
ca = certifi.where()


class NetworkSecurityException(Exception):
    def __init__(self, message, sys_obj):
        self.message = message
        self.sys_obj = sys_obj
        super().__init__(self.message)


class logging:
    @staticmethod
    def error(message):
        print(f"ERROR: {message}")

    @staticmethod
    def info(message):
        print(f"INFO: {message}")

class NetworkDataExtract:
    def __init__(self):
        try:
            logging.info("NetworkDataExtract object created.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        """
        Converts CSV data into JSON format.
        :param file_path: Path to the CSV file
        :return: List of JSON records
        """
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection, batch_size=1000):
        """
        Inserts JSON records into MongoDB in batches.
        :param records: List of JSON records to be inserted
        :param database: MongoDB database name
        :param collection: MongoDB collection name
        :param batch_size: Number of records to insert in each batch
        :return: Number of records inserted
        """
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca, serverSelectionTimeoutMS=50000, connectTimeoutMS=30000)
            db = self.mongo_client[database]
            col = db[collection]

            total_inserted = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    col.insert_many(batch)
                    total_inserted += len(batch)
                    logging.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")
                except Exception as batch_error:
                    logging.error(f"Batch insertion failed at batch {i // batch_size + 1}: {batch_error}")
            
            return total_inserted
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    try:
       
        FILE_PATH = "Network_Data/network_intrusion.csv"
        DATABASE = "Abhi"
        COLLECTION = "NetworkData"

      
        network_obj = NetworkDataExtract()

      
        records = network_obj.csv_to_json_convertor(file_path=FILE_PATH)
        logging.info(f"Total records to insert: {len(records)}")

       
        no_of_records = network_obj.insert_data_mongodb(records, DATABASE, COLLECTION)
        logging.info(f"Number of records successfully inserted: {no_of_records}")

    except NetworkSecurityException as nse:
        logging.error(f"Error occurred: {nse.message}")
