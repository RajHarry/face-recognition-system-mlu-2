from pymilvus import connections, FieldSchema, CollectionSchema, Collection, utility
from config import MILVUS_HOST, MILVUS_PORT

class MilvusOps:
    def __init__(self, collection_name) -> None:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection_name = collection_name
        self.collection = self.load_collection()

    # Load Mivus data in memory
    def load_collection(self):
        if(self.is_collection_exists()):
            collection = Collection(self.collection_name)      # Get an existing collection.
            collection.load()
            return collection
        else:
            raise Exception(f"Collection:'{self.collection_name}' not found!")

    def is_collection_exists(self):
        return utility.has_collection(self.collection_name)

    def remove_collection(self):
        return utility.drop_collection(self.collection_name)

    def add_to_collection(self, entities):
        return self.collection.insert(entities)

    def is_empid_exists(self, empid):
        res = self.collection.query(
            expr = f"emp_id in [{empid}]",
            offset = 0,
            limit = 1,
            consistency_level="Strong"
        )
        res_len = len(res)
        if(res_len == 1):
            return True
        return False

    def vector_search(self, vector):
        search_params = {"metric_type": "L2"}
        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=None,
            # set the names of the fields you want to retrieve from the search result.
            output_fields=['empId', 'empName'],
            consistency_level="Strong"
        )
        return results

    def close_connection(self):
        self.collection.release()

def create_collection(fields, collection_name):
    collection_schema = CollectionSchema(fields, segment_row_limit=4096, auto_id=False)
    return Collection(collection_name, collection_schema, consistency_level="Strong")

# milvus_obj = MilvusOps("employees")
# milvus_obj.remove_collection()