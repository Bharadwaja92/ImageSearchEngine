from qdrant_client import QdrantClient

# Connect to your Qdrant instance
# client = QdrantClient(host="qdrant-db", port=6333)
client = QdrantClient(host="localhost", port=6333)

collection_name = "images"
# snapshot_path = "file:///qdrant/storage/snapshots/images/QdrantSnapshot.snapshot"
snapshot_path = "QdrantSnapshot.snapshot"

client.recover_snapshot(
    collection_name=collection_name,
    location=snapshot_path,
)

print("Snapshot uploaded to Qdrant.")
