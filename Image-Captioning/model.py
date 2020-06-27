import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # Last layer truncated with -1
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_size=64, drop_prob=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_size
        self.n_layers = num_layers
        self.batch_size = batch_size
        self.embed_size = embed_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc2score = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state 
        self.hidden = self.init_hidden(batch_size)
        
    def init_hidden(self, batch_size):
        # The axes dimensions are (n_layers, batch_size, hidden_dim). batch_size explicitly made = 1
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())
    
    def forward(self, features, captions):
        # features shape: (batch_size, embed)
        # captions shape: (batch_size, seq)
        
        # Deleting the last sequence
        # new captions shape: (batch_size, [seq-1])
        captions = captions[:, :-1]
        
        # embeds shape: (batch_size, [seq-1], embed)
        embeds = self.word_embeddings(captions)
        
        # reshape features
        # new features shape: (batch_size, 1, embed)
        features = features.view(self.batch_size, 1, -1)
        
        # combine features and embed
        # input_tensor shape: (batch_size, seq, embed)
        input_tensor = torch.cat((features, embeds), dim=1)
        
        # lstm_out shape: (batch_size, seq, hidden_size)
        lstm_out, self.hidden = self.lstm(input_tensor, self.hidden)
        
        lstm_out = self.dropout(lstm_out)
        
        lstm_out_shape = lstm_out.shape        
        lstm_out_shape = list(lstm_out_shape)
        
        # new lstm_out shape: (batch_size*seq, hidden_size)
        lstm_out = lstm_out.view(lstm_out.size()[0]*lstm_out.size()[1], -1)
        
        # get the prob for the next word
        # vocab_outputs shape: (batch_size*seq, vocab_size)
        vocab_outputs = self.fc2score(lstm_out)
        
        # new vocab_outputs.shape: (batch_size, seq, vocab_size)        
        vocab_outputs = vocab_outputs.view(lstm_out_shape[0], lstm_out_shape[1], -1)
        
        return vocab_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ''' Given a character, predict the next character.
            Used for inference.
        
            Returns the predicted character and the hidden state.
        '''
        self.cuda()
                
        h = self.init_hidden(1)  # batch size of 1
        
        # inputs shape: (batch, seq, input), seq = 1       
        
        h = tuple([each.data for each in h])
        predicted_caption = []
        position_index = 0
        lstm_out = inputs
        pi_0 = 0
           
        for i in range(max_len):
        
            if pi_0 == 1:
                return predicted_caption
         
            
            # lstm_out shape: (batch_size, seq, hidden_size)
            lstm_out, h = self.lstm(lstm_out, h)

            lstm_out_shape = lstm_out.shape        
            lstm_out_shape = list(lstm_out_shape)

            # new lstm_out shape: (batch_size*seq, hidden_size)
            lstm_out = lstm_out.view(lstm_out.size()[0]*lstm_out.size()[1], -1)

            # get the prob for the next word
            # vocab_outputs shape: (batch_size*seq, vocab_size)
            lstm_out = self.fc2score(lstm_out)
            
            # Applying softmax
            p = F.softmax(lstm_out, dim=1).data

            _, temp_index = torch.max(p, 1)
            temp_index = temp_index.cpu()
            position_index = temp_index.numpy()
            pi_0 = position_index[0]
            predicted_caption.append(int(pi_0))
            position_index = position_index.reshape(lstm_out_shape[0], lstm_out_shape[1])
            position_index = torch.from_numpy(position_index)
            position_index = position_index.cuda()
        
            # embeds shape: (batch_size, seq, embed)
            lstm_out = self.word_embeddings(position_index)
            
        return predicted_caption