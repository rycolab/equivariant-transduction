import torch
from torch.nn.functional import softmax, log_softmax
from garchitecture import GConv, GEmbed, GDecode

class EquivariantHardAlignmentModel(torch.nn.Module):
    def __init__(self, input_vocab, output_vocab, G, K, num_filters, 
        hidden_size, num_layers, encode_embed_dim, decode_embed_dim, device, 
        nonlin = "tanh", training_max=False, annealed=False):
        super(EquivariantHardAlignmentModel, self).__init__()
        self.device = device
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.G = G
        self.K = K
        self.num_filters = num_filters
        self.gembed = GEmbed(self.G, self.K, self.input_vocab, 
            self.device).to(self.device)
        self.gconv = GConv(self.G, self.K, self.num_filters, 
            self.device).to(self.device)
        self.gdecode = GDecode(self.G, self.num_filters, self.output_vocab, 
            self.device).to(self.device)
        self.encode_embed_dim = encode_embed_dim
        self.encode_embed = torch.nn.Embedding(len(self.input_vocab), 
            self.encode_embed_dim)
        self.decode_embed_dim = decode_embed_dim
        self.decode_embed = torch.nn.Embedding(len(self.output_vocab), 
            self.decode_embed_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encode_LSTM = torch.nn.LSTM(self.encode_embed_dim, 
            self.hidden_size, self.num_layers, batch_first = True)
        self.decode_LSTM = torch.nn.LSTM(self.decode_embed_dim, 
            self.hidden_size, self.num_layers, batch_first = True)
        self.T = torch.nn.Parameter(torch.randn(self.hidden_size, 
            self.hidden_size * 2))
        torch.nn.init.uniform_(self.T, -0.5,0.5)
        self.out_bos_index = self.output_vocab.word_to_idx("<BOS>")
        self.out_eos_index = self.output_vocab.word_to_idx("<EOS>")
        self.nonlin = torch.tanh
        self.training_max = training_max
        self.annealed = annealed
    
    def forward(self, xs, ys, T=1):
        """
        Forward function to be used during training
        Input - xs: full input sequence (B x N_in x in_vocab_size x 1)
                ys: full correct output sequence 
                    (B x N_out x out_vocab_size x 1)
                T - temperature for annealing
        Output: log(p(ys | xs)) under the current model
        """

        # Pass input through G-equivariant stack to get output 
        # probabilities for all words in output vocab
        # (B x N_in x out_vocab_size)
        probs = self.gdecode(self.nonlin(self.gconv(self.nonlin(
            self.gembed(xs)))))
        # Get probabilities for output tokens in ys
        ys_idx = self.output_vocab.batch_tensor_to_sent(ys).to(self.device)
        encoder_scores = torch.gather(probs, 2, 
            ys_idx.unsqueeze(1).repeat(1, xs.shape[1], 1))

        # Use LSTM to encode input
        henc_s, h_0, c_0 = self.encode(xs)
        # And again in reverse to get bidirectional
        backward_xs = torch.flip(xs, dims=[1])
        hbackenc, _, _ = self.encode(backward_xs)
        hbackenc = torch.flip(hbackenc, dims=[1])

        # Get ys vocabulary indices
        idxs = self.output_vocab.batch_tensor_to_sent(ys)
        p = len(self.G)
        # Replace with representatives of lexical class
        idxs[(idxs<p).nonzero().transpose(0,1)[0,:], 
            (idxs<p).nonzero().transpose(0,1)[1,:]] = 0
        # Embed and encode with LSTM
        y_embed = self.decode_embed(idxs)
        output, hidden = self.decode_LSTM(y_embed, (h_0, c_0))
        h_0, c_0 = hidden

        # Concatenate final encoder hidden state and all decoder hidden
        # states
        h_dec = torch.cat([henc_s[:,-1,:].unsqueeze(1), output], 
            dim=1)[:, :-1,:]
        
        # Concatenate forward and backward encodings to get
        # bidirectional encoding for inputs
        h_enc = torch.cat((henc_s, hbackenc), dim=2)
        # Then alignment scores are from bilinear products of these 
        # states
        Th_enc = torch.matmul(
            self.T.unsqueeze(0).unsqueeze(0).repeat(xs.shape[0], 
                h_enc.shape[1],1,1), 
            h_enc.unsqueeze(-1)).squeeze(-1)
        eij = torch.matmul(
            h_dec.unsqueeze(1).repeat(1, h_enc.shape[1],1,1).unsqueeze(-2), 
            Th_enc.unsqueeze(2).repeat(1,1,h_dec.shape[1],1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
        alignment_scores = softmax(eij, dim=1)

        if not self.training_max:
            # Sum-based model
            p = torch.sum(torch.log(torch.sum(encoder_scores*alignment_scores, 
                dim=1)), dim=1)
        else:
            if self.annealed:
                # Annealed-max model
                scores = encoder_scores*alignment_scores
                annealing = torch.nn.functional.softmax((1/T)*scores, dim=1)
                p = torch.sum(torch.log(torch.sum(annealing*scores, dim=1)), 
                    dim=1)
            else:
                # Max model
                p = torch.sum(torch.log(torch.max(
                    encoder_scores*alignment_scores, dim=1)[0]), dim=1)
        p_total = torch.sum(p)
        return -1.0 * p_total
    
    def decode(self, xs, beam_size=5, max_length=100, use_max=False):
        """
        Take as input entire input sequence and decode output sequence 
        using beam search
        Input - xs: entire input sequence (1 x N_in x in_vocab_size x 1)
                beam_size: beam_size
                max_length: maximum length to generate
                use_max: deocde using max rather than sum
        Output - ys: predicted output sequence (1 x N_out)
        """
        # Get bidirectional LSTM encoding of input
        henc_s, h_enc, c_enc = self.encode(xs)
        backward_xs = torch.flip(xs, dims=[1])
        hbackenc, _, _ = self.encode(backward_xs)
        hbackenc = torch.flip(hbackenc, dims=[1])
        h_enc_all = torch.cat((henc_s, hbackenc), dim=2)
        torch.cuda.empty_cache()

        # For each x_i in xs, pre-calculate probabilities from the model
        # over  all possible output words
        encoder_scores = self.gdecode(self.nonlin(self.gconv(self.nonlin(
            self.gembed(xs)))))
        return self.beam_search(xs, h_enc_all, h_enc, c_enc, encoder_scores, 
            beam_size, max_length, use_max)
    
    def forward_arc(self, h_dec, h_enc, encoder_scores, use_max=False):
        """
        Forward function for use when decoding
        Inputs - h_dec: current hidden state of the decoder
                 c_dec: current cell state of the decoder
                 xs: full input sequence
                 h_enc: all hidden states from the encoder
                 hbackenc: all hidden states from encoder when run on 
                           reversed input sequence
                 y_t: most recent decoded output word
                 use_max: decode using max rather than sum
        Output - p: Log probabilities for next output word over entire 
                output vocabulary, to be used in beam search
        """
        # Get bilinear products of encoder and decoder states
        Th_enc = torch.matmul(
            self.T.unsqueeze(0).unsqueeze(0).repeat(h_enc.shape[0], 
                h_enc.shape[1],1,1), 
            h_enc.unsqueeze(-1)).squeeze(-1)
        eij = torch.matmul(
            h_dec.unsqueeze(1).repeat(1, h_enc.shape[1],1,1).unsqueeze(-2), 
            Th_enc.unsqueeze(2).repeat(1,1,h_dec.shape[1],1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
        alignment_scores = softmax(eij, dim=1)
        if not use_max:
            # Sum decoding
            p = torch.log(torch.sum(
                encoder_scores*alignment_scores.repeat(1,1,
                    encoder_scores.shape[-1]), dim=1))
        else:
            # Max decoding
            p = torch.log(torch.max(
                encoder_scores*alignment_scores.repeat(1,1,
                    encoder_scores.shape[-1]), dim=1)[0])
        return p

    def beam_search(self, xs, h_encs, h_enc, c_enc, encoder_scores, 
        beam_size=5,  max_length=100, use_max=False):
        """
        Perform beam search
        Inputs - xs: input sequence
                 h_encs: bidirectional encoding of xs
                 h_enc: initial decoder hidden state
                 c_enc: initial decoder cell state
                 encoder_scores: output of G-equivariant stack
                 beam_size: beam size
                 max_length: max length to decode
                 use_max: decode using max rather than sum
        Output - a decoded sentence
        """
        # Initialise beam: [sentence, logprob, done, 
        # current decoder hidden states, all decoder hidden states 
        # so far]
        beam = [[[], torch.tensor([0], dtype=torch.float32).to(self.device), 
                False, (h_enc, c_enc), h_enc.transpose(0,1)]]
        new_beam = []
        p = len(self.G)
        done = False
        while not done:
            new_beam = []
            for b in beam:
                # If sentence not finished
                if not b[2]:
                    h_0, c_0 = b[3]
                    h_dec = b[4]
                    # Get logprobs for next step
                    probs = self.forward_arc(h_dec, h_encs, encoder_scores,
                        use_max)
                    probs = probs.squeeze(0).to(self.device)
                    topk = torch.topk(probs, beam_size)
                    for w in topk.indices:
                        new_b = [b[0].copy(), 
                            b[1].clone().detach().to(self.device), b[2], (), 
                            b[4].clone().detach().to(self.device)]
                        new_b[1] += probs[w.item()]
                        new_b[0].append(w.item())
                        y_0 = w.item()
                        # Get next decoder hidden states
                        y_0_tens = self.output_vocab.batch_sent_to_tensor(
                            torch.tensor([y_0]).unsqueeze(0)).to(self.device)
                        idxs= self.output_vocab.batch_tensor_to_sent(
                            y_0_tens).to(self.device)
                        idxs[(idxs<p).nonzero().transpose(0,1)[0,:], 
                            (idxs<p).nonzero().transpose(0,1)[1,:]] = 0
                        y_embed = self.decode_embed(idxs)
                        output, hidden = self.decode_LSTM(y_embed, (h_0, c_0))
                        new_b[3] = hidden
                        new_b[4] = output
                        if w == self.out_eos_index or len(b[0]) > max_length:
                            new_b[2] = True
                        new_beam.append(new_b)
                else:
                    # Copy across finished sentences
                    new_b = [b[0].copy(), b[1].clone().detach().to(self.device),
                    b[2], b[3], b[4].clone().detach().to(self.device)]
                    new_beam.append(new_b)
            # Sort by logprob normalised by length
            new_beam.sort(key=lambda x:x[1]/len(x[0]), reverse=True)
            beam = new_beam[:beam_size]
            done = all([b[2] for b in beam])
        return torch.tensor(beam[0][0], dtype=torch.int64).to(self.device)

    def encode(self, xs):
        """
        Encode entire input sequence with invariant LSTM.
        Input - xs (B x N_in x in_vocab_size x 1)
        """

        # Initial hidden and cell states
        h_0 = torch.zeros((self.num_layers, xs.shape[0], self.hidden_size), 
            requires_grad=True).to(self.device)
        c_0 = torch.zeros((self.num_layers, xs.shape[0], self.hidden_size), 
            requires_grad=True).to(self.device)

        # Getting vocab indices
        idxs = self.input_vocab.batch_tensor_to_sent(xs).to(self.device)
        p = len(self.G)
        # Replacing words with lexical class representative
        idxs[(idxs<p).nonzero().transpose(0,1)[0,:], 
            (idxs<p).nonzero().transpose(0,1)[1,:]] = 0
        # Passing entire input sequence through embedding and Encoder 
        # LSTM and keeping hidden states
        x_embed = self.encode_embed(idxs)
        output, hidden = self.encode_LSTM(x_embed, (h_0, c_0))
        h_0, c_0 = hidden
        return output, h_0, c_0