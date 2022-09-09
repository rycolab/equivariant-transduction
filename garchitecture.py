import torch 

class GConv(torch.nn.Module):
    def __init__(self, G, K, num_filters, device):
        super(GConv, self).__init__()
        self.G = G
        self.num_filters = num_filters
        self.psi = torch.nn.Parameter(torch.randn(len(G), K, num_filters), 
            requires_grad=True)
        torch.nn.init.uniform_(self.psi, -0.5,0.5)
        self.K = K
        self.device = device
    
    def forward(self, f):
        """
        Input - f: output of G-Embed layer (B x N x |G| x K)
        Output - conv: G-Convolution of f (B x N x |G| x num_filters)
        """
        B = f.shape[0]
        N = f.shape[1]
        conv = torch.zeros(B, N, len(self.G), self.num_filters).to(self.device)
        n = len(self.G)
        # conv(f)_{g,d} = \sum_h f(h)⋅psi(g^{-1}h)_{d}
        for g in range(len(self.G)):
            for h in range(len(self.G)):
                conv[:, :, g, :] += torch.matmul(f[:,:,h,:].unsqueeze(2).repeat(
                    1, 1, self.num_filters, 1).unsqueeze(-2), 
                self.psi[(n - g + h) % n, :, :].transpose(0,1
                    ).unsqueeze(0).unsqueeze(0).repeat(B,N,1,1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
        return conv

class GEmbed(torch.nn.Module):
    def __init__(self, G, K, wordset, device):
        super(GEmbed, self).__init__()
        self.G = G
        self.psi = torch.nn.Parameter(torch.randn(len(wordset), K), 
            requires_grad=True)
        torch.nn.init.uniform_(self.psi, -0.5,0.5)
        self.K = K
        self.wordset = wordset
        self.device = device
    
    def forward(self, s):
        """
        Input - s: batch of sentences (B x N_in x in_vocab_size x 1)
        Output - E_w: a G-Embedding of the sentences 
                      (B x N_in x |G| x self.K)
        """
        B = s.shape[0]
        N = s.shape[1]
        E_w = torch.zeros(B, N, len(self.G), self.K).to(self.device)
        # e(x)_{g,k} = psi(g^{-1} x)_k
        for g in range(len(self.G)):
            E_w[:, :, g, :] = self.psi[self.wordset.batch_tensor_to_sent(
                self.G.act_inverse(g, s)) , :]
        return E_w

class GDecode(torch.nn.Module):
    def __init__(self, G, K, wordset, device):
        super(GDecode, self).__init__()
        self.G = G
        self.psi = torch.nn.Parameter(torch.randn(len(wordset), K), 
            requires_grad=True)
        torch.nn.init.uniform_(self.psi, -0.5,0.5)
        # gdecode.K = output dimension of GConv i.e. gconv.num_filters
        self.K = K
        self.wordset = wordset
        self.device = device
    
    def forward(self, phi):
        """
        Input - phi: Output of G-Convolution (B x N x |G| x num_filters)
        Output - sm: Probabilities over output vocabulary 
                     (B x N x out_vocab_size
        """
        B = phi.shape[0]
        N = phi.shape[1]
        logits = torch.zeros((B, N, len(self.wordset))).to(self.device)
        # logits(phi, w_out) = \sum_h phi(h)⋅psi(h^{-1}w_out)
        for h in range(len(self.G)):
            w_out = self.wordset.sent_to_tensor(torch.tensor(
                [i for i in range(len(self.wordset))]
                )).unsqueeze(0).to(self.device)
            hinv_w_out = self.wordset.batch_tensor_to_sent(
                self.G.act_inverse(h, w_out)).squeeze(0)
            toadd = torch.matmul(phi[:, :, h, :].unsqueeze(2).repeat(
                1,1,len(self.wordset),1).unsqueeze(-2), 
            self.psi[hinv_w_out, :].unsqueeze(0).unsqueeze(0).repeat(B,N,1,1
                ).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            logits[:,:,:] += toadd
        # Softmax to get probabilities
        sm = torch.nn.functional.softmax(logits, dim=2)
        return sm