import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from datasets import Dataset

class RNN(nn.Module): # highest working value for hidden size = 12_526
    def __init__(self, hidden_size=3132, output_size=19, EMBED_SIZE=768):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layers
        self.rnnL1 = nn.RNNCell(EMBED_SIZE, hidden_size, nonlinearity='tanh')
        self.rnnL2 = nn.RNNCell(hidden_size, EMBED_SIZE, nonlinearity='tanh')
        
        # Multi-head attention layers
        self.mha_rnn1 = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=False)
        self.mha_rnn2 = nn.MultiheadAttention(EMBED_SIZE, num_heads=1, batch_first=False)

        # Final classifier layer
        self.classL = nn.Linear(EMBED_SIZE, output_size)
        
        # BERT tokenizer and model for embeddings
        # It's good practice to load these once.
        self.bert_tokenizer = BertTokenizer.from_pretrained('./local-bert')
        self.emb_model = BertModel.from_pretrained('./local-bert')

        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(EMBED_SIZE)
        self.input_norm = nn.LayerNorm(EMBED_SIZE)

        # Dropout for stability
        self.dropout_attn = nn.Dropout(0.1)



    def tokenize_input(self, movie_ovrvw, device):
        """
        Tokenizes and embeds a single movie overview text.
        NOTE: This process is run outside the training loop to save memory.
        """
        tokenized_inp = self.bert_tokenizer.tokenize(text=movie_ovrvw)
        
        # The 'add_prefix' logic can be simplified
        processed_tokens = ['My', 'Sentence'] + tokenized_inp

        model_inputs = self.bert_tokenizer.encode_plus(
            processed_tokens,
            is_split_into_words=True,
            return_tensors='pt'
        ).to(device)

        # Use torch.no_grad() to get embeddings without tracking gradients
        # This is crucial for memory saving when you're just doing inference/embedding.
        with torch.no_grad():
            outputs = self.emb_model(**model_inputs, return_dict=True)
            
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings
    
    def forward(self, x):
        """
        Processes a batch of sequences through the RNN and attention layers.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, EMBED_SIZE)
        """
        # Get batch size and sequence length from the input tensor

        batch_size, T, _ = x.shape
        device = x.device

        # Normalize input
        x=self.input_norm(x)

        # Initialize hidden states - use zeros, not Xavier (Xavier is for weights, not activations)
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        h2 = torch.zeros(batch_size, 768, device=device)


        # First Pass: Collect hidden states
        h1_list = []
        h2_list = []

        for t in range(T):
            xt = x[:, t, :]  # Get the t-th timestep for all items in the batch
            
            # CRITICAL FIX: Removed .detach() here.
            # Detaching the hidden state in the loop prevents the gradient from flowing
            # back through time, which is essential for training an RNN.
            h1 = self.rnnL1(xt, h1)
            h1 = self.layer_norm1(h1)
            h2 = self.rnnL2(h1, h2)

            
            h1_list.append(h1)
            h2_list.append(h2)

        # Stack hidden states: shape changes from list of (batch, feat) to (seq_len, batch, feat)
        rnnL1_hidden_state_arr = torch.stack(h1_list, dim=0)
        rnnL2_hidden_state_arr = torch.stack(h2_list, dim=0)

        # --- Apply self-attention ---
        # The MHA layer expects (seq_len, batch, embed_dim)
        mhaL1, _ = self.mha_rnn1(rnnL1_hidden_state_arr, rnnL1_hidden_state_arr, rnnL1_hidden_state_arr)
        mhaL2, _ = self.mha_rnn2(rnnL2_hidden_state_arr, rnnL2_hidden_state_arr, rnnL2_hidden_state_arr)

        # Additional dropout on attention outputs (applied once, not per timestep)
        mhaL1 = self.dropout_attn(mhaL1)
        mhaL2 = self.dropout_attn(mhaL2)


        for t in range(T):
            # A single token (the input)
            xt = x[:, t, :] 
            
            # Use attention output as the context for the hidden state update
            ctx1 = mhaL1[t, :, :]  # Shape: the attention output of a single batch sample
            ctx2 = mhaL2[t, :, :]  # Shape: (batch, 768)

            h1 = self.rnnL1(xt, ctx1)
            h2 = self.rnnL2(h1, ctx2)

        # --- Final classification ---
        # The output is the last hidden state of the second pass
        logits = self.classL(h2)  # h2 has shape (batch, 768)
        
        return logits
