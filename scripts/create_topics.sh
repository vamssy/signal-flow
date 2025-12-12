#!/bin/bash
# Create Kafka topics for the stock prediction system

BOOTSTRAP_SERVER="${KAFKA_BOOTSTRAP_SERVER:-localhost:9092}"

echo "Creating Kafka topics on $BOOTSTRAP_SERVER..."

# Create topics with kafka-topics command (requires Kafka CLI tools installed)
# Or use docker exec if running in Docker

if command -v kafka-topics &> /dev/null; then
    kafka-topics --bootstrap-server $BOOTSTRAP_SERVER --create --if-not-exists --topic market_ohlcv --partitions 3 --replication-factor 1
    kafka-topics --bootstrap-server $BOOTSTRAP_SERVER --create --if-not-exists --topic features --partitions 3 --replication-factor 1
    kafka-topics --bootstrap-server $BOOTSTRAP_SERVER --create --if-not-exists --topic signals --partitions 3 --replication-factor 1
else
    echo "Using Docker to create topics..."
    docker exec stock-kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic market_ohlcv --partitions 3 --replication-factor 1
    docker exec stock-kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic features --partitions 3 --replication-factor 1
    docker exec stock-kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic signals --partitions 3 --replication-factor 1
fi

echo "Listing topics:"
if command -v kafka-topics &> /dev/null; then
    kafka-topics --bootstrap-server $BOOTSTRAP_SERVER --list
else
    docker exec stock-kafka kafka-topics --bootstrap-server localhost:9092 --list
fi

echo "Done!"

