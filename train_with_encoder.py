import numpy as np
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from transformer.Models import Transformer
import torch
import transformer.Constants as Constants
import tqdm 
import jieba 
import MeCab
import langid 
import os
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import thulac

thul = thulac.thulac(seg_only=True)
mecab = MeCab.Tagger("-Owakati") 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def read_instances_from_file(inst_file, max_word_seq_len):
    
    words_insts = []
    trimmed_sent_count = 0
    labels = []
    with open(inst_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            sent = line.strip()
            label = int(sent.split('\t')[0])
            inst = sent.split('\t')[1]
            #print(inst)
            inst = inst.replace(' ','')
            sent_tuple = langid.classify(inst)
            if sent_tuple[0] == 'zh':
                text = thul.cut(inst, text=True)
                words = text.split(' ')
            elif sent_tuple[0] == 'ja':
                words = (mecab.parse(sent)).split()
            if len(words) > max_word_seq_len + 1:
                trimmed_sent_count += 1
            words_inst = words[:max_word_seq_len]
            
            if words_inst and label:
                words_insts += [[Constants.BOS_WORD] + words_inst + [Constants.EOS_WORD]]
                labels.append(label)
            else:
                words_insts += [None]
                labels.append(label)
    print('[Info] Get {} instances from {}'.format(len(words_insts), inst_file))
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_word_seq_len))

    return words_insts, labels
    
def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def collate_fn(insts, labels):
    ''' Pad the instance to the max seq length in batch '''
    
    max_len = max(len(inst) for inst in insts)
    
    #batch_seq = np.array([
    #    inst + [Constants.PAD] * (max_len - len(inst))
    #    for inst in insts])
    
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    
    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)
    #labels = torch.from_numpy(labels)
    labels = torch.LongTensor(labels)
    
    return batch_seq, batch_pos, labels
    
def prepare_dataloader(features, labels, batch_size):
    ''' Turn the dataset into dataloader'''
#     features = np.array(features)
#     labels = np.array(labels)
#     tensor_dataset = TensorDataset(torch.from_numpy(features),torch.from_numpy(labels))
    batch_seq, batch_pos,labels = collate_fn(features,labels)
    tensor_dataset = TensorDataset(batch_seq, batch_pos, labels)
    dataloader = DataLoader(tensor_dataset, num_workers=2,
                            batch_size=batch_size)
    return dataloader
    
class Extract_encoder(object):
    '''Load the encoder of the transformer'''
    
    def __init__(self,model):
        
        self.device = torch.device('cuda')
        checkpoint = torch.load(model)
        checkpoint_copy = checkpoint['model'].copy()
        
        for k in list(checkpoint_copy.keys()):
            new_key = k.replace('module.model.','')
            checkpoint_copy.update({str(new_key):checkpoint_copy.pop(k)})

        model_opt=checkpoint['settings']
        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)
        model.load_state_dict(checkpoint_copy)
        model = model.to(self.device)
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        
    def encoder_out(self,src_seq,src_pos):
    
        enc_output = self.model.encoder(src_seq,src_pos)
        
        return enc_output
        
class Multi_model(nn.Module):
    '''Transformer added LSTM'''
    def __init__(self,Extract_encoder):
        super().__init__()
        self.use_gpu = True
        self.batch_size = 16
        self.bidirection = False
        self.encoder = Extract_encoder
        self.embedding = nn.Embedding(133829, 300, padding_idx=Constants.PAD)
        self.num_layers = 1
        self.hidden_size = 256
        self.LSTM = nn.LSTM(
                input_size=300 + 512,
                hidden_size=256,
                num_layers=1,
                bidirectional=False,
                )
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(256, 105)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc.weight)

        
    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.bidirection:
            direction_coef = 2 
        else :
            direction_coef = 1
        
        if self.use_gpu:
            
            hidden = Variable(torch.zeros(direction_coef * self.num_layers, 
                batch_size,
                self.hidden_size)).cuda()#*direction_coef)).cuda()
            cell = Variable(torch.zeros(direction_coef * self.num_layers,
                batch_size,
                self.hidden_size )).cuda()#* direction_coef)).cuda()
        else :
            
            hidden = Variable(torch.zeros(direction_coef * self.num_layers,
                batch_size,
                self.hidden_size * direction_coef))
            cell = Variable(torch.zeros(direction_coef * self.num_layers,
                batch_size,
                self.hidden_size*direction_coef))
        return (hidden,cell)        
    
    def forward(self,src_seq, src_pos):
    
        encoder_out = self.encoder.encoder_out(src_seq, src_pos)
        
        embedding_out = self.embedding(src_seq)
        
        all_features = torch.cat([encoder_out[0], embedding_out],2)
        
        all_features = all_features.permute(1,0,2)
        
        self.hidden = self.init_hidden(src_seq.size()[0])
        
        lstm_out, self.hidden = self.LSTM(all_features,self.hidden)
        
        lstm_out = lstm_out[-1]
        
        fc_out = self.fc(lstm_out)
        
        fc_out = self.dropout(fc_out)
        
        return fc_out 
        
def turn_labels(labels):
    '''Turn the labels begin from 0'''
    
    label_dict = {}
    label_dict[4] = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1] and labels[i] not in label_dict:
            label_dict[labels[i]] = len(label_dict)
    new_label = []
    for j in labels:
        new_label.append(label_dict[j])
    return label_dict, new_label 
    
def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data', required=True)
    parser.add_argument('-test_data', required=True)
    parser.add_argument('-max_word_seq_len', type=int, default=30)
    parser.add_argument('-model',required=True)
    parser.add_argument('-vocab',required=True)
    parser.add_argument('-co_training', action='store_true')
    parser.add_argument('-epochs', type=int, default = 50)
    parser.add_argument('-encoder_lr', default=0.0001)
    parser.add_argument('-lr',default=0.001)
    opt = parser.parse_args()
	
    #opt.max_token_seq_len = opt.max_word_seq_len + 3
    
    train_sec_word_insts, labels = read_instances_from_file(opt.train_data, opt.max_word_seq_len)
    test_sec_word_insts, test_labels = read_instances_from_file(opt.test_data, opt.max_word_seq_len)
    
    label_dict, labels = turn_labels(labels)
    
    preprocess_data = torch.load(opt.vocab)
    word_dict = preprocess_data['dict']['src']
    
    train_insts = convert_instance_to_idx_seq(train_sec_word_insts, word_dict)
    test_insts = convert_instance_to_idx_seq(test_sec_word_insts, word_dict)
    
    new_test_labels = []
    for label in test_labels:
        new_test_labels.append(label_dict[label])
    
    np.random.seed(13)
    np.random.shuffle(train_insts)
    np.random.seed(13)
    np.random.shuffle(labels)
    np.random.seed(13)
    np.random.shuffle(test_insts)
    np.random.seed(13)
    np.random.shuffle(new_test_labels)
    
    training_loader = prepare_dataloader(train_insts, labels, batch_size=64)
    test_loader  = prepare_dataloader(test_insts,new_test_labels,batch_size=64)
    
    encoder = Extract_encoder(opt.model)
    multi_model = Multi_model(encoder).cuda()
    
    criterion = nn.CrossEntropyLoss()
    if opt.co_training:
        encoder_params = list(map(id, multi_model.encoder.model.parameters()))
        base_paramas = filter(lambda p : id(p) not in encoder_params, multi_model.parameters())
        optimizer = optim.Adam([
            {'params': base_paramas, 'lr': opt.lr},
            {'params': multi_model.encoder.model.parameters(), 'lr':opt.encoder_lr}
            ])
    else:
        optimizer = optim.Adam(filter(lambda p : p.requires_grad, multi_model.parameters()), lr = 0.001)
    
    counter = 0
    clip = 5
    train_losses = []
    multi_model.train()
    for epoch in range(opt.epochs):
        test_losses = []
        pred_labels = []
        true_labels = []
        pred_acc = []
        num_correct = 0
        for batch in training_loader:
            counter += 1 
            sec_seq, sec_pos, y = batch
            sec_seq, sec_pos,y = sec_seq.cuda(), sec_pos.cuda(), y.cuda()
            multi_model.zero_grad()
            output = multi_model.forward(sec_seq, sec_pos)
            loss = criterion(output.squeeze(), y)
            train_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(multi_model.parameters(), clip)
            optimizer.step()
            if counter % 50 == 0:
                print("Epoch: {}/{}...".format(epoch+1, opt.epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(np.mean(train_losses)))
                train_losses = []
                      #"Val Loss: {:.6f}".format(np.mean(val_losses)))
        multi_model.eval()
        for batch in test_loader:
            sec_seq, sec_pos, y = batch
            sec_seq, sec_pos, y = sec_seq.cuda(), sec_pos.cuda(), y.cuda()
            pred = multi_model.forward(sec_seq, sec_pos)
            loss = criterion(pred.squeeze(), y)
            test_losses.append(loss.item())
            pred = pred.data.max(1)[1]
            pred_labels += list(pred.cpu().numpy())
            true_labels += list(y.cpu().numpy())
            correct_tensor = pred.eq(y).view_as(pred)
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / len(test_loader.dataset)
        pred_acc.append(test_acc)
        if test_acc > max(pred_acc):
            torch.save(multi_model.model_state_dict, 'tc_best.chkpt')
            print('the best model has been updated')
        print("Test accuracy: {:.3f}".format(test_acc))  
        adjust_learning_rate(optimizer)
        multi_model.train()
        
if __name__ == '__main__':
    main()