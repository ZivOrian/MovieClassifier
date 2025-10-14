import torch
from torch import nn
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from datasets import Dataset


device = "cuda:0" if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, hidden_size=1462):
        super(RNN, self).__init__()
        # Continue model design further down here:
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax()
        self.rnnL1 = nn.RNNCell(768, hidden_size, nonlinearity='tanh')
        self.rnnL2 = nn.RNNCell(hidden_size, 768, nonlinearity='tanh')
        self.classL = nn.Linear(768,19) # The final, classifier layer
        self.mha_rnn1 = torch.nn.MultiheadAttention(hidden_size,1)
        self.mha_rnn2 = torch.nn.MultiheadAttention(768,1)
        self.h1 = torch.zeros(1, hidden_size,device=device).detach()    # shape (batch, hidden)
        self.h2 = torch.zeros(1, 768,device=device).detach()               # shape (batch, 768)
        
        
        #Redacted code kept in case the bert model and tokenizer need to be downloaded again
        """self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emb_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer.save_pretrained('./local-bert')
        self.emb_model.save_pretrained('./local-bert')"""

        self.bert_tokenizer = BertTokenizer.from_pretrained('./local-bert')
        self.emb_model = BertModel.from_pretrained('./local-bert')
        

    # A function for Dataset.map()
    def add_prefix(self,example):
        example['tokens'] = ['My', 'Sentence'] + example['tokens']
        return example
    

    def tokenize_input(self, movie_ovrvw, device):
        tokenized_inp = self.bert_tokenizer.tokenize(text=movie_ovrvw)
        # Convert the list of tokens into a Dataset object
        dataset = Dataset.from_dict({'tokens': [tokenized_inp]})
        processed_tokens = dataset.map(self.add_prefix)['tokens'][0]

        # Encode the input
        model_inputs = self.bert_tokenizer.encode_plus(
            processed_tokens,
            is_split_into_words=True,
            return_tensors='pt'
        ).to(device)

        # Use torch.no_grad() to get embeddings without tracking gradients
        with torch.no_grad():
            outputs = self.emb_model(**model_inputs, return_dict=True)
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings
    
    
    def forward(self, x):
        """
        x: iterable/sequence of length T where each x[t] is shape (768,) or (1,768).
        We'll treat batch size = 1 for simplicity (you can generalize).
        """
        #device = x[0].device if isinstance(x, (list, tuple)) else x.device
        T = len(x)

        # --- initialize local hidden states (batch_size=1 here) ---
       

        # --- collect hidden states per timestep for attention ---
        h1_list = []
        h2_list = []


        # Running the RNN without the single head attention
        for t in range(T):
            # ensure x[t] is (1, 768)
            xt = x[t].view(1, -1)
            self.h1 = self.rnnL1(xt, self.h1).detach()
            self.h2 = self.rnnL2(self.h1, self.h2).detach()
            h1_list.append(self.h1)  # keep the computational graph
            h2_list.append(self.h2)

        # Stack into tensors with shape (seq_len, batch, embed) as expected by nn.MultiheadAttention
        # resulting shapes: (T, 1, hidden_size) and (T, 1, 768)
        rnnL1_hidden_state_arr = torch.stack(h1_list, dim=0).detach()
        rnnL2_hidden_state_arr = torch.stack(h2_list, dim=0).detach()

        # Applying self attention on the hidden units of rnn for every time step
        mhaL1, _ = self.mha_rnn1(rnnL1_hidden_state_arr, rnnL1_hidden_state_arr, rnnL1_hidden_state_arr)
        mhaL2, _ = self.mha_rnn2(rnnL2_hidden_state_arr, rnnL2_hidden_state_arr, rnnL2_hidden_state_arr)

        # --- THIS IS THE second pass using attention-context as the "hidden" for the next RNN run ---
        # start from fresh local hidden states again (or reuse h1/h2 if you want continuity)

        # Running the RNN with the single head attention
        for t in range(T):
            xt = x[t].view(1, -1)
            # mhaL1[t] has shape (1, embed), if it's (1,1,embed) use squeeze:
            ctx1 = mhaL1[t].view(1, -1)  # (1, hidden_size)
            ctx2 = mhaL2[t].view(1, -1)  # (1, 768)

            self.h1 = self.rnnL1(xt, ctx1)    # use attention output as hidden/state input
            self.h2 = self.rnnL2(self.h1, ctx2)


        del h1_list, h2_list, rnnL1_hidden_state_arr, rnnL2_hidden_state_arr, mhaL1, mhaL2# Deletion to free up GPU space
        # final classifier on last hidden
        logits = self.classL(self.h2.squeeze(0))   # classL expects (batch, feat); here batch=1
        
        torch.cuda.empty_cache()
        return logits
