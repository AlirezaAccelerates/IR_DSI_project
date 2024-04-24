import random
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from tqdm.notebook import tqdm

# Precision at K for TF-IDF
def precision_at_k_tfidf(queries, doc_ids, tfidf_matrix, vectorizer, k, num_queries=5):
    '''
    queries: Dictionary of queries with their embeddings and correlated doc IDs.
    doc_ids: List of document IDs.
    tfidf_matrix: TF-IDF matrix.
    vectorizer: TF-IDF vectorizer.
    k: Number of top documents to consider for calculating precision.
    num_queries: Number of queries to consider.
    '''
    # Initialize the results dictionary
    results = {}

    # Get num_queries random queries
    query_ids = random.sample(list(queries.keys()), num_queries)

    for query_id in tqdm(query_ids, desc=f"Computing Precision at {k}"):
        # Get the relevant documents for the query
        relevant_docs = queries[query_id]["docids_list"]
        # Compute the TF-IDF matrix
        query_tfidf = vectorizer.transform([queries[query_id]['raw'].lower()])
        # Compute cosine similarities
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        # Get the top K retrieved documents
        top_k_retrieved = [doc_ids[i] for i in np.argsort(cosine_similarities, axis=0)[::-1]][:k]
        # Count how many of the top K retrieved documents are relevant
        relevant_count = sum(doc_id in relevant_docs for doc_id in top_k_retrieved)
        # Return the precision at K
        results[query_id] = relevant_count / k

    return results


# Extract top k docids for a given query
def top_k_docids_siamese(model, query_info, documents, k):    
    '''
    model: The trained model.
    query_info: Dictionary containing query information.
    documents: Dictionary containing document information.
    k: Number of top documents to return.
    ret_type: Type of embedding to use for computing the similarity score.
    '''
    # Determine the return type
    if model.__class__.__name__ == 'SiameseNetwork': ret_type = 'emb'
    if model.__class__.__name__ == 'SiameseTransformer': ret_type = 'first_L_emb'
    if model.__class__.__name__ == 'SiameseNetworkPL': ret_type = 'emb'

    # Get the query embedding
    query_emb = torch.tensor(query_info[ret_type], dtype=torch.float32)
    
    # Initialize list for storing scores
    scores = []

    # Iterate through each document
    for doc_id, doc_info in tqdm(documents.items(), desc=f"Computing Top {k} DocIDs"):
        # Get the document embedding
        doc_emb = torch.tensor(doc_info[ret_type], dtype=torch.float32)
        
        # If the model is a SiameseNetwork
        if model.__class__.__name__ == 'SiameseNetwork':
            # Compute the score
            score = model(query_emb.unsqueeze(-1).permute(1,0).to(model.device) , doc_emb.unsqueeze(-1).permute(1,0).to(model.device)).item()

        # If the model is a SiameseTransformer
        elif model.__class__.__name__ == 'SiameseTransformer':
            # Pad the embeddings if necessary
            query_emb_padded = F.pad(query_emb.unsqueeze(0), (0, model.embedding_size - query_emb.size(0))).squeeze(0)
            doc_emb_padded = F.pad(doc_emb.unsqueeze(0), (0, model.embedding_size - doc_emb.size(0))).squeeze(0)
            # Compute the score
            score = model(query_emb_padded.unsqueeze(-1).permute(1,0).to(model.device), doc_emb_padded.unsqueeze(-1).permute(1,0).to(model.device)).item()
        
        # If the model is a SiameseNetworkPL
        elif model.__class__.__name__ == 'SiameseNetworkPL':
            # Compute the score
            score = F.cosine_similarity(model(query_emb).unsqueeze(0), model(doc_emb).unsqueeze(0)).item()

        # Append the score to the list
        scores.append([doc_id, score])

    # Rank documents based on scores comouted by the model and extract top k docids
    top_k_docids = np.array(sorted(scores, key=lambda x: x[1], reverse=True))[:k, 0]
    
    # Return top k docids
    return top_k_docids


# Compute top k docids greedly
def top_k_docids(model, query, k, max_length, decode_docid_fn, docid_sos_token=12):
    '''
    model: The trained seq2seq model.
    query: The input query as a tensor (batch_size, seq_len).
    k: The number of sequences to return.
    max_length: The maximum length of each generated sequence.
    docid_sos_token: The ID of the start token.
    '''
    # Ensure the query tensor has a batch dimension
    if query.dim() == 1:
        query = query.unsqueeze(1)  # Add batch dimension
    elif query.size(0) < query.size(1):
        query = query.permute(1, 0)  # Swap dimensions if necessary

    # Initialize the input tensor with the start token for each item in the batch
    input_seq = torch.full((1, k), docid_sos_token, dtype=torch.long, device=query.device)

    # Initialize a tensor to store the top k sequences
    top_k_sequences = torch.zeros(max_length, k, dtype=torch.long, device=query.device)

    for t in range(max_length):
        output = model(query.repeat(1, k), input_seq)  # Repeat query for k hypotheses
        next_token_probs, next_tokens = torch.topk(output[-1, :, :], k, dim=-1)

        if t == 0:
            # For the first step, all k tokens are different
            # Select the top 1 token from each of the k hypotheses
            top_k_sequences[t] = next_tokens[0]
        else:
            # For subsequent steps, choose the next token based on the highest probability
            # Use argmax to find the indices of the highest probability tokens
            selected_indices = next_token_probs.argmax(dim=-1)
            top_k_sequences[t] = next_tokens.gather(1, selected_indices.unsqueeze(0)).squeeze()

        # Update input_seq for the next step with the selected tokens
        input_seq = torch.cat((input_seq, top_k_sequences[t].unsqueeze(0)), dim=0)

    # Convert the top k sequences to a numpy array of decoded ids
    top_k_sequences = np.array([''.join(map(str, decode_docid_fn(lst))) for lst in top_k_sequences.t().tolist()])

    return top_k_sequences


# Constrained beam search
def top_k_beam_search(model, query, trie_data, k, max_length, decode_docid_fn, docid_sos_token=12, docid_eos_token=10):
    '''
    model: PyTorch model used to compute the next token probabilities.
    query: Query tensor.
    trie_data: Trie data structure.
    k: Number of top sequences to return.
    max_length: Maximum length of the sequences.
    decode_docid: Function used to decode the doc ID sequences.
    docid_sos_token: Start of sequence token.
    docid_eos_token: End of sequence token.
    '''
    # Handle query tensor batch dimension
    if query.dim() == 1:
        query = query.unsqueeze(0)
    
    # Permute batch dimension
    query = query.permute(1, 0)

    # Initialize the hypotheses with log prob = 0 (since log(1) = 0)
    hypotheses = [{'tokens': [docid_sos_token], 'log_prob': 0}]

    for _ in range(max_length):
        # Initialize list for storing all possible candidates
        all_candidates = []

        # Iterate through each hypothesis
        for h in hypotheses:
            # Skip extension for completed sequences
            if h['tokens'][-1] == docid_eos_token:
                all_candidates.append(h)
                continue
            
            # Get the input sequence tensor
            input_seq = torch.tensor(h['tokens'], dtype=torch.long, device=query.device).unsqueeze(1)
            # Get the output from the model
            output = model(query, input_seq)
            # Get possible next tokens
            next_tokens = trie_data.get_next_tokens(h['tokens'])
            # Get probabilities for the next tokens
            log_probs = torch.log_softmax(output[-1, 0, next_tokens], dim=0)

            # Create new hypotheses and add them to the list
            for idx, token in enumerate(next_tokens):
                new_hyp = {'tokens': h['tokens'] + [token], 'log_prob': h['log_prob'] + log_probs[idx].item()}
                all_candidates.append(new_hyp)

        # Keep top k hypotheses
        hypotheses = sorted(all_candidates, key=lambda x: x['log_prob'], reverse=True)[:k]

    # Extract token sequences
    top_k_sequences = np.array([''.join(map(str, decode_docid_fn(h['tokens']))) for h in hypotheses])

    return top_k_sequences


# Average precision
def compute_AP(top_k_ids, docids_list):
    '''
    top_k_ids: List of top k retrieved document IDs.
    docids_list: List of relevant document IDs.
    '''
    # Create a boolean mask where top_k_ids are in docids_list
    is_relevant = np.isin(top_k_ids, docids_list)
    # If there are no relevant documents, return 0
    if np.all(~is_relevant): return 0
    # Generate an array of positions (1-indexed)
    positions = np.arange(1, len(top_k_ids) + 1)
    # Compute precision at each relevant position
    precision_at_relevant = np.where(is_relevant, 1 / positions, 0)
    # Filter out non-zero elements and compute the mean
    non_zero_mean = precision_at_relevant[precision_at_relevant != 0].mean()
    # Return
    return non_zero_mean


def compute_RAK(top_k_ids, docids_list):
    '''
    top_k_ids: List of top k retrieved document IDs.
    docids_list: List of relevant document IDs.
    '''
    return np.sum(np.isin(top_k_ids, docids_list)) / len(docids_list)


def compute_PAK(top_k_ids, docids_list):
    '''
    top_k_ids: List of top k retrieved document IDs.
    docids_list: List of relevant document IDs.
    '''
    return np.isin(top_k_ids, docids_list).mean()


# Mean average precision
def compute_Mean_metrics(model, test_queries, queries, documents, trie_data=None, dataset=None, k=10, max_length=10, model_type='seq2seq'):
    '''
    model: PyTorch model used to compute the next token probabilities.
    trie_data: Trie data structure.
    dataset: Dataset object.
    queries: Dictionary of queries.
    k: Number of top sequences to return.
    max_length: Maximum length of the sequences.
    '''
    # Initialize running mean average precision
    running_mean_AP = 0.0
    running_mean_PatK = 0
    running_mean_RatK = 0

    # Iterate over test dataset
    for i, query in enumerate(tqdm(test_queries, desc="Computing Mean Metrics")):
        # Get the list of relevant docids
        

        # Compute top-k docids for the current query
        if model_type == 'seq2seq':
            docids_list = np.array(queries[dataset.query_ids[query]]['docids_list'])
            top_k_ids = np.array(top_k_beam_search(model, query, trie_data, k=k, max_length=max_length, decode_docid_fn=dataset.decode_docid))
        elif model_type == 'siamese':
            docids_list = np.array(queries[query]['docids_list'])
            top_k_ids = top_k_docids_siamese(model, queries[query], documents, k=1000)

        # Compute average precision for the current query
        current_AP = compute_AP(top_k_ids[:k], docids_list)
        # Compute the precision at k
        current_PAK = compute_PAK(top_k_ids[:k], docids_list)
        # Compute the recall at k
        current_RAK = compute_RAK(top_k_ids, docids_list)

        # Update the running mean
        running_mean_AP = running_mean_AP + (current_AP - running_mean_AP) / (i+1)
        running_mean_PatK = running_mean_PatK + (current_PAK - running_mean_PatK) / (i+1)
        running_mean_RatK = running_mean_RatK + (current_RAK - running_mean_RatK) / (i+1)

    # Return the running mean average precision
    return running_mean_AP, running_mean_PatK, running_mean_RatK
