import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel

class RNN(nn.Module):
    def __init__(self, hidden_size=2350, output_size=19, EMBED_SIZE=768):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.EMBED_SIZE = EMBED_SIZE
        
        # RNN layers
        self.rnnL1 = nn.RNNCell(EMBED_SIZE, hidden_size, nonlinearity='tanh')
        self.rnnL2 = nn.RNNCell(hidden_size, EMBED_SIZE, nonlinearity='tanh')
        
        # Multi-head attention layers
        self.mha_rnn1 = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=False)
        self.mha_rnn2 = nn.MultiheadAttention(EMBED_SIZE, num_heads=1, batch_first=False)

        # Final classifier layer
        self.classL = nn.Linear(EMBED_SIZE, output_size)
        
        # BERT tokenizer and model for embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('./local-bert')
        self.emb_model = BertModel.from_pretrained('./local-bert')

        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(EMBED_SIZE)
        self.input_norm = nn.LayerNorm(EMBED_SIZE)

        # Dropout for stability
        self.dropout_attn = nn.Dropout(0.1)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def tokenize_input(self, movie_ovrvw, device):
        """
        Tokenizes and embeds a single movie overview text.
        """
        tokenized_inp = self.bert_tokenizer.tokenize(text=movie_ovrvw)
        processed_tokens = ['My', 'Sentence'] + tokenized_inp

        model_inputs = self.bert_tokenizer.encode_plus(
            processed_tokens,
            is_split_into_words=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = self.emb_model(**model_inputs, return_dict=True)
            
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings
    
    def forward(self, x, seq_lengths):
        """
        Processes a batch of sequences through the RNN and attention layers.
        NOW PROPERLY HANDLES PADDING!
        
        Args:
            x (Tensor): Padded input tensor of shape (batch_size, max_seq_len, EMBED_SIZE)
            seq_lengths (Tensor): Actual lengths of each sequence (batch_size,)
        """
        batch_size, max_T, _ = x.shape
        device = x.device

        # Normalize input
        x = self.input_norm(x)

        # Initialize hidden states with ZEROS
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        h2 = torch.zeros(batch_size, self.EMBED_SIZE, device=device)

        # First Pass: Collect hidden states
        h1_list = []
        h2_list = []

        # Process only up to actual sequence lengths (not padding)
        for t in range(max_T):
            xt = x[:, t, :]
            
            # Create mask for valid timesteps (not padding)
            mask = (t < seq_lengths).float().unsqueeze(1).to(device)  # (batch, 1)
            
            # Process timestep
            h1_new = self.rnnL1(xt, h1)
            h1_new = self.layer_norm1(h1_new)
            h2_new = self.rnnL2(h1_new, h2)
            h2_new = self.layer_norm2(h2_new)
            
            # ✅ KEY FIX: Only update hidden states where mask is 1 (not padding)
            # For padded positions, keep the previous hidden state
            h1 = mask * h1_new + (1 - mask) * h1
            h2 = mask * h2_new + (1 - mask) * h2
            
            h1_list.append(h1)
            h2_list.append(h2)

        # Stack hidden states
        rnnL1_hidden_state_arr = torch.stack(h1_list, dim=0)  # (seq_len, batch, hidden)
        rnnL2_hidden_state_arr = torch.stack(h2_list, dim=0)  # (seq_len, batch, embed)

        # Apply self-attention
        mhaL1, _ = self.mha_rnn1(rnnL1_hidden_state_arr, rnnL1_hidden_state_arr, rnnL1_hidden_state_arr)
        mhaL2, _ = self.mha_rnn2(rnnL2_hidden_state_arr, rnnL2_hidden_state_arr, rnnL2_hidden_state_arr)

        # Dropout on attention outputs
        mhaL1 = self.dropout_attn(mhaL1)
        mhaL2 = self.dropout_attn(mhaL2)

        # Reset hidden states for second pass
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        h2 = torch.zeros(batch_size, self.EMBED_SIZE, device=device)

        # Second pass with attention context
        for t in range(max_T):
            xt = x[:, t, :] 
            ctx1 = mhaL1[t, :, :]
            ctx2 = mhaL2[t, :, :]

            # Create mask for valid timesteps
            mask = (t < seq_lengths).float().unsqueeze(1).to(device)

            h1_new = self.rnnL1(xt, ctx1)
            h1_new = self.layer_norm1(h1_new)
            h2_new = self.rnnL2(h1_new, ctx2)
            h2_new = self.layer_norm2(h2_new)
            
            # Only update where not padding
            h1 = mask * h1_new + (1 - mask) * h1
            h2 = mask * h2_new + (1 - mask) * h2

        # Final classification using the last valid hidden state
        # h2 now contains the correct final state (not influenced by padding)
        logits = self.classL(h2)
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-10, max=10)
        
        return logits