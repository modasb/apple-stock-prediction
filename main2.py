from confluent_kafka import Consumer, KafkaError
import json
from datetime import datetime, timedelta
import time
import logging
import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB settings
mongo_host = 'localhost'
mongo_port = 27017

# Kafka settings
KAFKA_TOPIC = 'stock_prices'

# Connect to MongoDB
try:
    client = MongoClient(mongo_host, mongo_port)
    db = client['stock_data']
    collection = db['stock_prices']
    collection.create_index([('symbol', 1), ('timestamp', 1)], unique=True)
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

def consume_messages():
    # Configure Kafka consumer
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'stock_consumer_group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Consumer subscribed to topic: {KAFKA_TOPIC}")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info('Reached end of partition')
                else:
                    logger.error(f'Error: {msg.error()}')
            else:
                try:
                    record = json.loads(msg.value().decode('utf-8'))
                    collection.update_one(
                        {"symbol": record['symbol'], "timestamp": record['timestamp']},
                        {"$set": record},
                        upsert=True
                    )
                    logger.info(f"Processed record: {record['timestamp']}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        consumer.close()
        client.close()

if __name__ == "__main__":
    try:
        consume_messages()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Consumer terminated")
