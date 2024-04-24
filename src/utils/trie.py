from typing import List


class TrieNode:
    def __init__(self):
        # Each node has a dictionary of children nodes.
        self.children = {}
        # Each node has a boolean flag to indicate if it is the end of a valid sequence.
        self.is_end_of_sequence = False

class Trie:
    def __init__(self, pad_token_id=11):
        self.pad_token_id = pad_token_id
        # Initialize the trie with a root node.
        self.root = TrieNode()

    # Insert a sequence (without <sos>, <eos>, or <pad>) into the trie.
    def insert(self, sequence: List[int]):
        # Start from the root node.
        node = self.root

        # For each token in the sequence:
        for token in sequence:
            # If the token is <pad>, stop.
            if token == self.pad_token_id:
                break  
            # If the token is not in the children of the current node, add it.
            if token not in node.children:
                node.children[token] = TrieNode()
            # Move to the next node.
            node = node.children[token]

        # Mark the end of a valid sequence.
        node.is_end_of_sequence = True 

    # Check if a sequence (without <sos>, <eos>, or <pad>) is in the trie.
    def check_sequence(self, sequence: List[int]):
        # Start from the root node.
        node = self.root

        # For each token in the sequence:
        for token in sequence:
            # If the token is not in the children of the current node, return False.
            if token not in node.children:
                return False
            # Move to the next node.
            node = node.children[token]

        # Return True if the current node is the end of a valid sequence.
        return node.is_end_of_sequence

    # Get all possible next tokens given a prefix sequence (without <sos>, <eos>, or <pad>).
    def get_next_tokens(self, prefix_sequence: List[int]):
        # Start from the root node.
        node = self.root

        # For each token in the prefix sequence:
        for token in prefix_sequence:
            # If the token is not in the children of the current node, return an empty list.
            if token not in node.children:
                return []
            # Move to the next node.
            node = node.children[token]

        # Return all the children of the current node.
        return list(node.children.keys())

    # Count the number of sequences in the trie.
    def _count_sequences(self, node):
        # Initialize the count.
        count = 0
        # If the current node is the end of a valid sequence, increment the count.
        if node.is_end_of_sequence:
            count += 1
        # Recursively count the number of sequences for each child node.
        for child in node.children.values():
            count += self._count_sequences(child)
        # Return the count.
        return count

    def __len__(self):
        # Call the recursive helper function starting from the root node.
        return self._count_sequences(self.root)
