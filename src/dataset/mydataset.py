import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from tqdm.notebook import tqdm
from src.utils import dataset_utils

# (query, document, relevance) dataset class
class QueryDocumentDataset(Dataset):
    '''
    queries (dict): Dictionary with query information.
    documents (dict): Dictionary with document information.
    ret_type (str): Return type. Can be 'id', 'raw', 'emb', or 'first_L_emb'.
    '''

    def __init__(
            self, 
            queries: dict, 
            documents: dict, 
            ret_type: str ='id'
        ):

        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.data = self.build_dataset(queries, documents)

    @staticmethod
    def build_dataset(queries, documents):
        dataset = []
        # Create a set of all document IDs
        all_doc_ids = set(documents.keys())
        
        # For each query
        for query_id, query_data in tqdm(queries.items(), desc='Building QueryDocumentDataset'):
            # Get the list of correlated documents
            docid_list = set(query_data['docids_list'])
            # Add positive examples
            for doc_id in docid_list:
                dataset.append((query_id, doc_id, 1))

            # Add negative examples, randomly sampled from all_doc_ids excluding docid_list
            for doc_id in set(random.sample(all_doc_ids - docid_list, len(docid_list))):
                dataset.append((query_id, doc_id, 0))

        return dataset
    
    def set_ret_type(self, new_ret_type):
        self.ret_type = new_ret_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the query ID, document ID and relevance label
        query_id, doc_id, relevance = self.data[idx]

        # If the return type is 'id', return the IDs
        if self.ret_type == 'id':
            return query_id, doc_id, relevance
        
        # If the return type is 'raw', return the raw text
        elif self.ret_type == 'raw':
            return self.queries[query_id]['raw'], self.documents[doc_id]['raw'], relevance
        
        # If the return type is 'emb', return the embeddings
        elif self.ret_type == 'emb':
            return  torch.tensor(self.queries[query_id]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[doc_id]['emb'], dtype=torch.float32), \
                    torch.tensor(relevance, dtype=torch.float32)
        
        # If the return type is 'first_L_emb', return the first L tokens embeddings
        elif self.ret_type == 'first_L_emb':
            return  torch.tensor(self.queries[query_id]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[doc_id]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(relevance, dtype=torch.float32)
    

# (query, document (positive), document (negative)) dataset class
class TripletQueryDocumentDataset(Dataset):
    '''
    queries (dict): Dictionary with query information.
    documents (dict): Dictionary with document information.
    ret_type (str): Return type. Can be 'id', 'raw', 'emb', or 'first_L_emb'.
    '''
    def __init__(
            self, 
            queries: dict, 
            documents: dict, 
            ret_type: str ='id'
        ):
        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.triplets = self.build_dataset(queries, documents)

    @staticmethod
    def build_dataset(queries, documents):
        # Initialize the dataset list
        dataset = []
        # Create a set of all document IDs
        all_doc_ids = set(documents.keys())

        # For each query
        for query_id, query_data in tqdm(queries.items(), desc='Building TripletQueryDocumentDataset'):
            # Get the list of correlated documents
            positive_docs = set(query_data['docids_list'])
            # Get the list of negative documents, randomly sampled from all_doc_ids excluding positive_docs
            negative_docs = random.sample(all_doc_ids - set(positive_docs), len(positive_docs))
            # Add positive and negative examples
            for positive_doc, negative_doc in zip(positive_docs, negative_docs):
                dataset.append((query_id, positive_doc, negative_doc))
        
        # Return the dataset
        return dataset
    
    def set_ret_type(self, new_ret_type):
        self.ret_type = new_ret_type

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Get the query ID, positive document ID and negative document ID
        anchor, positive, negative = self.triplets[idx]
        
        # If the return type is 'id', return the IDs
        if self.ret_type == 'id':
            return anchor, positive, negative
        
        # If the return type is 'raw', return the raw text
        if self.ret_type == 'raw':
            return self.queries[anchor]['raw'], self.documents[positive]['raw'], self.documents[negative]['raw']
        
        # If the return type is 'emb', return the embeddings
        if self.ret_type == 'emb':
            return  torch.tensor(self.queries[anchor]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[positive]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[negative]['emb'], dtype=torch.float32)
        
        # If the return type is 'first_L_emb', return the first L tokens embeddings
        if self.ret_type == 'first_L_emb':
            return  torch.tensor(self.queries[anchor]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[positive]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[negative]['first_L_emb'], dtype=torch.float32)


# (encoded_doc, encoded_docid) dataset class
class DocumentDataset(Dataset):
    '''
    documents (dict): Dictionary with document information.
    doc_max_len (int): Maximum length of the input sequence.
    '''
    def __init__(
            self, 
            documents: dict, 
            doc_max_len: int = 32
        ):

        self.documents = documents
        self.doc_max_len = doc_max_len

        # Semantic docid initialization
        self.semantic_to_docid = None
        self.semantic = False
        
        # We use the T5 tokenizer to encode the documents
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Define the docid special tokens
        self.docid_eos_token = 10       # End of string token for docids
        self.docid_pad_token = 11       # Padding token for docids
        self.docid_sos_token = 12     # Start of string token for docids

        # Compute the maximum docid length
        self.docid_max_len = max(len(str(docid)) for docid in documents.keys()) + 2  # for EOS token and start token
        self.semantic_docid_max_len = None # To be set later

        # Initialize the encoded documents and docids lists
        self.encoded_docs, self.docids = self.build_dataset()
        # Initialize the semantic docids list
        self.semantic_docids = []
    
    def build_dataset(self):
        # Initialize the encoded documents and docids lists
        docids = []
        encoded_docs = []

        # For each document in the documents dictionary
        for docid, content in tqdm(self.documents.items(), desc='Building DocumentDataset'):
            # Tokenizing and encoding the document text
            preprocessed_text = dataset_utils.preprocess_text(content['raw'])
            preprocessed_text = " ".join(preprocessed_text)
            encoded_doc = self.tokenizer.encode(preprocessed_text,
                                           add_special_tokens=True,
                                           max_length=self.doc_max_len,
                                           truncation=True)

            # Padding the document sequence to max_length
            encoded_doc = F.pad(torch.tensor(encoded_doc), (0, self.doc_max_len - len(encoded_doc)), value=0)

            # Encoding the docid (treating each character as a digit)
            encoded_docid = torch.tensor([self.docid_sos_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                    [self.docid_pad_token] * (self.docid_max_len - len(docid)))

            # Appending the encoded document and docid to the lists
            docids.append(encoded_docid)
            encoded_docs.append(encoded_doc)

        return encoded_docs, docids
    
    def set_semantic(self, semantic):
        self.semantic = semantic

    def add_semantic_docids_mapping(self, semantic_docids_mapping, semantic_docid_max_len):
        # Add the semantic docids mapping
        self.semantic_to_docid = semantic_docids_mapping
        # Set the semantic docid max length
        self.semantic_docid_max_len = semantic_docid_max_len
        # Iterate over the semantic docids
        for docid in self.semantic_to_docid:
            semantic_docid = torch.tensor([self.docid_sos_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                            [self.docid_pad_token] * (self.semantic_docid_max_len - len(docid)))
            # Add the semantic docid to the list
            self.semantic_docids.append(semantic_docid)

    def __len__(self):
        return len(self.encoded_docs)

    def __getitem__(self, idx):
        return self.encoded_docs[idx], self.docids[idx] if not self.semantic else self.semantic_docids[idx]

    def decode_docid(self, encoded_docid):
        # Convert to list (if it's not already)
        encoded_docid_list = encoded_docid.tolist() if isinstance(encoded_docid, torch.Tensor) else encoded_docid

        # Remove the start token if it's the first character
        if encoded_docid_list[0] == self.docid_sos_token:
            encoded_docid_list = encoded_docid_list[1:]

        # Check if EOS token is in the list and keep only the part of the list up to the EOS token
        if self.docid_eos_token in encoded_docid_list:
            encoded_docid_list = encoded_docid_list[:encoded_docid_list.index(self.docid_eos_token)]
        else:
            # If EOS token is not found, handle accordingly (e.g., use the entire list or throw an error)
            encoded_docid_list = [self.docid_eos_token]

        # Convert the remaining tokens to string and join them
        decoded = ''.join(map(str, encoded_docid_list))

        return decoded if not self.semantic else self.semantic_to_docid.get(decoded, '10')
    

# (encoded_query, encoded_docid) dataset class
class RetrievalDataset(Dataset):
    '''
    documents (dict): Dictionary with document information.
    queries (dict): Dictionary with query information.
    query_max_len (int): Maximum length of the input sequence.
    '''
    def __init__(
            self, 
            documents: dict, 
            queries: dict, 
            query_max_len: int = 9
        ):

        self.documents = documents
        self.queries = queries
        self.query_max_len = query_max_len

        # Semantic docid initialization
        self.semantic_to_docid = None
        self.semantic = False

        # Initialize tokenized query to query dictionary
        self.query_ids = dict()
        
        # We use the T5 tokenizer to encode the documents
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Define the docid special tokens
        self.docid_eos_token = 10   # End of string token for docids
        self.docid_pad_token = 11   # Padding token for docids
        self.docid_sos_token = 12   # Start of string token for docids

        # Compute the maximum docid length
        self.docid_max_len = max(len(str(docid)) for docid in documents.keys()) + 2  # for EOS token and start token
        self.semantic_docid_max_len = None # To be set later

        # Initialize the encoded documents and docids lists
        self.encoded_queries, self.docids =  self.build_dataset()
        # Initialize the semantic docids list
        self.semantic_docids = []

    def build_dataset(self):
        # Initialize the encoded documents and docids lists
        docids = []
        encoded_queries = []

        # Iterate over the queries
        for queryid, content in tqdm(self.queries.items(), desc='Building RetrievalDataset'):
            # Tokenizing and encoding the document text
            preprocessed_text = dataset_utils.preprocess_text(content['raw'])
            preprocessed_text = " ".join(preprocessed_text)
            encoded_query = self.tokenizer.encode(preprocessed_text,
                                                add_special_tokens=True,
                                                max_length=self.query_max_len,
                                                truncation=True)

            # Padding the document sequence to max_length
            encoded_query = F.pad(torch.tensor(encoded_query), (0, self.query_max_len - len(encoded_query)), value=0)
            
            # Add the tokenized query to query dictionary
            self.query_ids[encoded_query] = queryid

            # For each document in the documents dictionary
            for docid in content['docids_list']:
                # Encoding the docid (treating each character as a digit)
                encoded_docid = torch.tensor([self.docid_sos_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                        [self.docid_pad_token] * (self.docid_max_len - len(docid)))


                #self.docids.append(encoded_docid)
                encoded_queries.append(encoded_query)
                docids.append(encoded_docid)

        return encoded_queries, docids
    
    def set_semantic(self, semantic):
        self.semantic = semantic

    def add_semantic_docids_mapping(self, semantic_docids_mapping, semantic_docid_max_len):
        # Add the semantic docids mapping
        self.semantic_to_docid = semantic_docids_mapping
        # Set the semantic docid max length
        self.semantic_docid_max_len = semantic_docid_max_len
        # Iterate over the semantic docids
        for docid in self.semantic_to_docid:
            semantic_docid = torch.tensor([self.docid_sos_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                            [self.docid_pad_token] * (self.semantic_docid_max_len - len(docid)))
            # Add the semantic docid to the list
            self.semantic_docids.append(semantic_docid)

    def __len__(self):
        return len(self.encoded_queries)

    def __getitem__(self, idx):
        return self.encoded_queries[idx], self.docids[idx] if not self.semantic else self.semantic_docids[idx]

    def decode_docid(self, encoded_docid):
        # Convert to list (if it's not already)
        encoded_docid_list = encoded_docid.tolist() if isinstance(encoded_docid, torch.Tensor) else encoded_docid

        # Remove the start token if it's the first character
        if encoded_docid_list[0] == self.docid_sos_token:
            encoded_docid_list = encoded_docid_list[1:]

        # Check if EOS token is in the list and keep only the part of the list up to the EOS token
        if self.docid_eos_token in encoded_docid_list:
            encoded_docid_list = encoded_docid_list[:encoded_docid_list.index(self.docid_eos_token)]
        else:
            # If EOS token is not found, error in generating the docid, so we return the EOS token
            encoded_docid_list = [self.docid_eos_token]

        # Convert the remaining tokens to string and join them
        decoded = ''.join(map(str, encoded_docid_list))

        return decoded if not self.semantic else self.semantic_to_docid.get(decoded, '10')