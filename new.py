'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import os
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class Compute_Loss(nn.Module):
    
    def __init__(self,Constants,smoothing):
        
        super(Compute_Loss,self).__init__()
        self.Constants = Constants
        self.smoothing = smoothing
        
    def cal_performance(self,pred,gold,smoothing=False):
        
        gold = gold.contiguous().view(-1)#.cuda()
        loss = self.cal_loss(pred,gold,smoothing)
        pred = pred.max(1)[1]
        non_pad_mask = gold.ne(Constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum()
        
        return loss,n_correct
    
    def cal_loss(self,pred,gold,smoothing):
        gold = gold.contiguous().view(-1)
        
        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(Constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

        return loss

def train_epoch(model, training_data, optimizer, smoothing):
    
    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        _, _, tgt_seq, _ = batch
        gold = tgt_seq[:, 1:]
        loss, n_correct = model(batch)
        loss = loss.sum()
        n_correct = n_correct.sum()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total 
    accuracy = n_word_correct/n_word_total
    return loss_per_word,accuracy         

def eval_epoch(model,validation_data):
    
    model.eval()
    
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            _, _, tgt_seq, _ = batch
            gold = tgt_seq[:, 1:]
            loss,n_correct = model(batch)
            loss = loss.sum()
            n_correct = n_correct.sum()
            total_loss += loss.item()
            
            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            
        loss_per_word = total_loss / n_word_total 
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy

def train(model,training_data,validation_data,optimizer,opt):
    '''Strat  training'''
    
    log_train_file = None
    log_valid_file = None
    smoothing=opt.label_smoothing
    
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')
            
    valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        
        start = time.time()
        train_loss ,train_accu = train_epoch(
            model, training_data, optimizer ,smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))
        
        #print('  - (Training)   ppl: {ppl: 8.5f},elapse: {elapse:3.3f} min'.format(
                 # ppl=math.exp(min(train_loss, 100)), #accu=100*train_accu,
                 # elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model,validation_data)
        print('  - (Validation) ppl: {ppl: 8.5f},accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))
        #print('  - (Validation) ppl: {ppl: 8.5f},elapse: {elapse:3.3f} min'.format(
         #           ppl=math.exp(min(valid_loss, 100)),elapse=(time.time()-start)/60))            
        valid_accus += [valid_accu]
        valid_losses += [valid_loss]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model':model_state_dict,
            'settings':opt,
            'epoch':epoch_i}
        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                #if valid_accu >= max(valid_accus):
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

class FullModel(nn.Module):
    
    def __init__(self,model,loss):#,optimizer):
        super(FullModel,self).__init__()
        self.model =  model
        self.loss = loss
        #self.optimizer = optimizer
    
    def forward(self,batch):
        #print(self.model)
        #print(self.loss)
        src_seq, src_pos, tgt_seq, tgt_pos = batch
        gold = tgt_seq[:, 1:]
        #self.optimizer.zero_grad()
        pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
        loss, n_correct = self.loss.cal_performance(pred,gold)
        #print(torch.unsqueeze(loss,0))
        return torch.unsqueeze(loss,0),n_correct                    


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    
    parser.add_argument('-load_model',action='store_true')
    parser.add_argument('-model',default='trained.chkpt',
                        help=' load the model to train bidirection')
    
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size
    
    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)
    
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)#.to(device)
    
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    
    smoothing=opt.label_smoothing
    compute_loss = Compute_Loss(Constants,smoothing)
    model = FullModel(transformer,compute_loss)#,optimizer)
    model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
    
    train(model,training_data,validation_data,optimizer,opt)
    
    
def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return train_loader, valid_loader

    
if __name__ == '__main__':
    main()