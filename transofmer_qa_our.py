# Main src: https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
import torch
import torch.nn as nn
import torch.optim as optim
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        # (masked_fill): Wherever the mask is 0, replace the attention score with -1e9.
        # Why -1e9 ? Because immediately afterward you do: attn_probs = torch.softmax(attn_scores, dim=-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):  # [5, 8, 128]
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        # Divide the 128 into 2 heads: 2 x 64
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)  # [5, 8, 2, 64]
        x = x.transpose(1, 2)  # [5, 2, 8, 64]
        return x

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        '''
        What does x.transpose(1, 2) do?
            It swaps dimensions 1 and 2 of tensor x.        
        BUT: This is a view, not a copy. 
            It just reinterprets the strides, not the actual memory layout.
        Some PyTorch operations (like .view()) require the data to be in contiguous memory.
            Transpose can break that by creating a non-contiguous tensor.

            What does .contiguous() do? It creates a new tensor that has the same data but stored in contiguous memory.
        '''
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):  # Q.shape: [5, 8, 128]
        # Apply linear transformations and split heads
        # W_q and others don't change input shape
        Q = self.split_heads(self.W_q(Q))  # heads = 2: [5, 2, 8, 64]
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.unsqueeze(0) is used to add a batch dimension to the positional encoding.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


'''
Independence of Positions: While the self-attention mechanism processes the sequence 
    elements in a way that considers their contextual relationships, 
    the FFNN treats each position independently. This means that for a given layer, 
    the FFNN processes each element of the sequence in isolation, without considering the other elements.

Nonlinearity: The FFNN introduces nonlinearity into the model, which is crucial for 
    the model’s ability to learn complex patterns. Relu is used 

Dimensionality: The input and output of the FFNN typically have the same dimensionality 
    to ensure consistency across layers in the transformer. However, the inner-layer has 
    a higher dimensionality to allow the network to capture more complex features.
'''

# Feed Forward in Transformer
'''
Because after attention has gathered information from other words, 
we need a neural network to process and transform the resulting representation.
'''

'''
Attention says:

"Bring me information from other words."

Feed-forward says:

"Now that I have this information, let me process it."
'''

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # Trick in code is this is (batch, word, dimension)
        # When we do matrix multiplication, the batch multiplication
        # will only be applied on EACH SINGLE position
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

'''
Encoder
mask is usually for padding
Suppose your input is:
"I love cats <PAD> <PAD>"

The encoder should not let real words attend to <PAD> tokens.

So the mask might look conceptually like:

        I  love  cats  PAD  PAD
I       ✓   ✓     ✓    ✗    ✗
love    ✓   ✓     ✓    ✗    ✗
cats    ✓   ✓     ✓    ✗    ✗
PAD     ✗   ✗     ✗    ✗    ✗
PAD     ✗   ✗     ✗    ✗    ✗
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # src_mask: Source mask to ignore certain parts of the encoder's output.
    # tgt_mask: Target mask to ignore certain parts of the decoder's input.
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)  # [5, 9, 128]
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)  # Q, K, V
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)  # [5, 9, 128]
        x = self.norm3(x + self.dropout(ff_output))
        return x  # [5, 9, 128]


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,
                 num_heads, num_layers, d_ff, max_seq_length, dropout, pad_token_id):
        super(Transformer, self).__init__()
        self.pad_token_id = pad_token_id
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        '''
        remember:
        the masking are multiplied by natrix after calculating scores , which 
        has shape (batch , heads , seq_len , seq_len), so src_mask must have shape (batch_size, 1, 1, src_seq_len),
        then broadcating automatically expenses (batch_size, 1, 1, src_seq_len) across all heads and query positions, 
        sensing every token ignores padding positions.
        '''
        src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        '''
        (tgt != self.pad_token_id)
        Shape: (batch_size, tgt_seq_len)
        .unsqueeze(1) .unsqueeze(3)then:
        Inserts dimensions at index1(for heads) and index3(for key sequence length).
        Shape: (batch_size, 1, tgt_seq_len, 1)
        nopeak:creates a lower-triangular matrix of True values so position $i$can only attend to positions 
        Shape: (1, tgt_seq_len, tgt_seq_len)
        merging:
        result: (batch_size,1,seq_len,seq_len)
        '''
        tgt_mask = (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        '''
        The problem masks solve

        Attention computes a score for "how much should query token X pay attention to key token Y". 
        These scores form a grid (matrix): rows = queries, columns = keys.

        We use masks to force some of those scores to be ignored (set to -infinity before softmax, so they become 0 attention).

        Encoder (source) mask — same rule for every row

        Say your source sentence is 4 tokens long, but only the first 3 are real words — token 4 is padding (<pad>).

        The rule is simple: nobody is allowed to look at token 4, no matter who's asking.

        So the "allowed" grid looks like this (1 = allowed, 0 = blocked), for a 4x4 score matrix:

                key1 key2 key3 key4(pad)
        query1:  1    1    1    0
        query2:  1    1    1    0
        query3:  1    1    1    0
        query4:  1    1    1    0

        Every row is identical. Since every row is the same, you don't need to store 4 rows — you just store one row: [1, 1, 1, 0],
        and let it apply to all queries automatically. 
        That's why the shape has a 1 where the query dimension would be: (batch, 1, 1, src_len). The "1"s mean "broadcast/copy this across everything else."

        Decoder (target) mask — different rule per row

        In the decoder, there's an extra rule: token X can only look at tokens up to and including itself (no cheating by seeing future words).

        So for a 4-token target sequence, the grid looks like:

                key1 key2 key3 key4
        query1:  1    0    0    0     <- token1 can only see itself
        query2:  1    1    0    0     <- token2 can see 1,2
        query3:  1    1    1    0     <- token3 can see 1,2,3
        query4:  1    1    1    1     <- token4 can see everyone

        Now every row is different. Row 1 ≠ Row 2 ≠ Row 3 ≠ Row 4. You can't compress this into one row anymore — you genuinely need the full seq_len x seq_len grid. 
        That's why the shape is (batch, 1, seq_len, seq_len) instead of (batch, 1, 1, seq_len).

        '''
    def forward(self, src, tgt):
        # src_mask is for a padding mask to avoid attention over meaningless padded tokens

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


######################################################
# Following code is not from the top link


class SimpleTokenizer:
    def __init__(self, vocab, max_seq_length=512):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.token_to_id = {token: id for id, token in enumerate(vocab)}
        self.id_to_token = {id: token for token, id in self.token_to_id.items()}

    def encode(self, text):
        tokens = text.split()
        token_ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
        return token_ids[:self.max_seq_length]

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        return ' '.join(tokens)


def pad_sequences(sequences, max_seq_length, padding_value):
    # Pad all sequences to match the longest one
    padded_sequences = [seq + [padding_value] * (max_seq_length - len(seq)) for seq in sequences]
    return padded_sequences


def answer_me(model, question, tokenizer, max_seq_length):
    model.eval()
    device = next(model.parameters()).device

    src_tokens = tokenizer.encode(question)
    src = torch.tensor([src_tokens], device=device)

    tgt_tokens = [tokenizer.bos_token_id]
    tgt = torch.tensor([tgt_tokens], device=device)

    for _ in range(max_seq_length):
        output = model(src, tgt)
        next_token_logits = output[:, -1, :]
        # Or you can sample based on the probability as we did in RNN
        next_token_id = next_token_logits.argmax(dim=-1).item()

        # Break if end-of-sequence token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token_id)
        tgt = torch.tensor([tgt_tokens], device=device)

    # Decode the token IDs to a string
    del tgt_tokens[0]  # remove starting token
    generated_sequence = tokenizer.decode(tgt_tokens)
    return generated_sequence


if __name__ == '__main__':
    # Define the questions and answers. We have 5 Qs here
    questions = [
        "What is your name ?",
        "How old are you ?",
        "What is your profession ?",
        "Where do you live ?",
        "What do you teach ?"
    ]
    answers = [
        "My name is Moustafa Saad Ibrahim",
        "I am 37 years old",
        "I am a computer vision software engineer",  # longest of 7 words
        "I live in California since 2023",
        "teaching deep learning"
    ]
    # tokens for begin of sequence, end of sequence and unknown words
    bos_token, eos_token, unknown_token, pad_token = '<STS>', '<EOS>', '<UNK>', '<PAD>'
    # add BOS in each answer start to trigger the decoder first time
    answers = [bos_token + ' ' + answer for answer in answers]
    # add EOS for answer and question
    answers = [answer + ' ' + eos_token for answer in answers]
    questions = [question + ' ' + eos_token for question in questions]

    # Build the vocabulary from the text data
    all_text = questions + answers
    unique_words = set(word for sentence in all_text for word in sentence.split())
    unique_words = unique_words | set([bos_token, eos_token, unknown_token, pad_token])

    vocab = list(unique_words)
    tokenizer = SimpleTokenizer(vocab=vocab)
    # hack to add this info to the tokenizer
    tokenizer.bos_token_id = tokenizer.token_to_id[bos_token]
    tokenizer.eos_token_id = tokenizer.token_to_id[eos_token]
    tokenizer.unknown_token_id = tokenizer.token_to_id[unknown_token]
    tokenizer.pad_token_id = tokenizer.token_to_id[pad_token]

    tokenized_questions = [tokenizer.encode(question) for question in questions]
    tokenized_answers = [tokenizer.encode(answer) for answer in answers]

    # padding. In practice, we pad question and answer seperately
    max_seq_length = max(len(seq) for seq in tokenized_questions + tokenized_answers)
    tokenized_questions = pad_sequences(tokenized_questions, max_seq_length, tokenizer.pad_token_id)
    tokenized_answers = pad_sequences(tokenized_answers, max_seq_length, tokenizer.pad_token_id)

    # In language translation, src and target have different vocabulary
    src_vocab_size = len(tokenizer.token_to_id)
    tgt_vocab_size = len(tokenizer.token_to_id)
    d_model = 128
    num_heads = 2
    num_layers = 2
    d_ff = 265
    dropout = 0.1
    n_epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_data = torch.tensor(tokenized_questions, device=device)   # (batch_size=#Qs, seq_length)
    tgt_data = torch.tensor(tokenized_answers, device=device)

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers,
                              d_ff, max_seq_length, dropout, tokenizer.pad_token_id)
    transformer.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    warmup_steps = 4000
    optimizer = optim.Adam(transformer.parameters(), lr=1e-9, betas=(0.9, 0.98), eps=1e-9)
    def lr_lambda(step):
        return (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    lrs = []
    for epoch in range(n_epochs):
        transformer.train()
        optimizer.zero_grad()

        # Each word in target should help predict next one, except last (no next)
        target_input = tgt_data[:, :-1]     # 5 x 8
        # tgt_data[:, 1:] is a shifted 5x8 to match each one with next as GT
        target_output = tgt_data[:, 1:].reshape(-1)     # 40. Flat for loss

        predict_output = transformer(src_data, target_input)        # [5, 8, 38]. 38 is vocab size
        predict_output = predict_output.reshape(-1, tgt_vocab_size) # 40 x 38: shape: [batch_size * seq_len, vocab_size]

        loss = criterion(predict_output, target_output)
        loss.backward()
        optimizer.step()

        scheduler.step()  # 🔁 update learning rate (should be after each batch not epochs)
        lrs.append(optimizer.param_groups[0]['lr'])
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # you can now visualize the LR curve
    # import matplotlib.pyplot as plt
    # plt.plot(lrs)
    # plt.title("Learning Rate over Steps")
    # plt.xlabel("Step")
    # plt.ylabel("LR")
    # plt.show()

    transformer.eval()
    val_src_data = src_data
    val_tgt_data = tgt_data

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size),
                             val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")

    for question in questions:
        answer = answer_me(transformer, question, tokenizer, max_seq_length)
        print(question, ' ==> ', answer)

'''
Epoch: 97, Loss: 3.9350736141204834
Epoch: 98, Loss: 3.8883609771728516
Epoch: 99, Loss: 3.9018747806549072
Epoch: 100, Loss: 3.9631199836730957
Validation Loss: 3.909310817718506
What is your name ? <EOS>  ==>  What ? a you teaching learning a 2023 learning
How old are you ? <EOS>  ==>  What is software Where in <PAD> 2023 learning 2023
What is your profession ? <EOS>  ==>  What ? a old software 37 <PAD> 2023 learning
Where do you live ? <EOS>  ==>  in <PAD> Moustafa are in <PAD> 2023 learning a
What do you teach ? <EOS>  ==>  in <PAD> Moustafa are in <PAD> 2023 learning a

'''
