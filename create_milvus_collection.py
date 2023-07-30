from pymilvus import DataType, FieldSchema, connections
from utils.milvus_ops import create_collection

collection_name = "employees"
connections.connect(host='localhost', port='19530')
image_dimensions = 512

fields=[
        FieldSchema('empId', DataType.VARCHAR, is_primary=True, max_length=10),
        FieldSchema('empName', DataType.VARCHAR, max_length=50),
        FieldSchema('embedding', DataType.FLOAT_VECTOR, dim=image_dimensions),
        FieldSchema('timestamp', DataType.INT64)
    ]
print(f"Create collection `{collection_name}`")
collection = create_collection(fields, collection_name)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 4096
    }
}

collection.create_index(
  field_name="embedding",
  index_params=index_params,
  index_name="embs"
)