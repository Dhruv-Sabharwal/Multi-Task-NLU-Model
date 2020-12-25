import math
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#!pip install transformers
#!pip install ftfy
#!pip install spacy
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, AdamW
%load_ext tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/BaselineModel')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').to(device)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<system>", "<user>", "<slots>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<system>', '<user>', '<slots>']}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

add_special_tokens_(model, tokenizer)

def prepare_data(data_name, data_type):

    with open('C:/Users/dhruv/Desktop/ASP sem 1/Capstone/Conversational model/data/'+data_type+'_'+data_name+'_en.json') as f:
        data = json.load(f)
        
    # Breaking data into list of strings
    conversations = []
    for i in range(len(data)):
        conv = ['<system> Hello , welcome to the automated restaurant system . How may I help you ?']
        for j in range(len(data[i]['dialogue'])):
            conv.append('<user> '+data[i]['dialogue'][j]['transcript'])
            if(j<len(data[i]['dialogue'])-1):
                text = '<slots> '
                for k in range(len(data[i]['dialogue'][j]['turn_label'])):
                    text += data[i]['dialogue'][j]['turn_label'][k][0] + ' '
                    text += data[i]['dialogue'][j]['turn_label'][k][1] + ' '
                conv.append(text+'<system> '+data[i]['dialogue'][j+1]['system_transcript'])
            if (j == len(data[i]['dialogue'])-1):
                conv.append('<slots> <system> Have a nice day.')
            conversations.append(conv[:])

    # Adding <bos> and <eos> tokens to the sentences
    for i in range(len(conversations)):
        conversations[i][0] = SPECIAL_TOKENS[0]+' '+conversations[i][0]
        conversations[i][len(conversations[i])-1] = conversations[i][len(conversations[i])-1]+' '+SPECIAL_TOKENS[1]

    # Tokenizing input sentences            
    conversations_tokenized = []
    for i in range(len(conversations)):
        conv = []
        for j in range(len(conversations[i])):
            conv.append(tokenizer(conversations[i][j])['input_ids'])
        lens = 0
        for k in range(len(conv)):
            lens += len(conv[k])
        if(lens <= 400):  # maximum length of conversation = 400
            conversations_tokenized.append(conv)

    # Generating token_type_ids as a single list for each conversation
    token_type_ids = []
    for i in range(len(conversations_tokenized)):
        tokens = []
        for j in range(len(conversations_tokenized[i])):
            if(j%2==0):
                for k in range(len(conversations_tokenized[i][j])):
                    tokens.append('<system>')
            else:
                for k in range(len(conversations_tokenized[i][j])):
                    tokens.append('<user>')
        token_type_ids.append(tokens)

    # Tokenizing token_type_ids
    for i in range(len(token_type_ids)):
        token_type_ids[i] = tokenizer.convert_tokens_to_ids(token_type_ids[i]) 

    # Generating tokenized lm_labels
    lm_labels = []
    for i in range(len(conversations_tokenized)):
        label = []
        for j in range(len(conversations_tokenized[i])-1):
            for k in range(len(conversations_tokenized[i][j])):
                label.append(-100)
        label.append(-100)
        for k in range(len(conversations_tokenized[i][-1])-1):
            label.append(conversations_tokenized[i][-1][k+1])
        lm_labels.append(label)

    # Generating input_ids
    input_ids = []
    for i in range(len(conversations_tokenized)):
        tokens = []
        for j in range(len(conversations_tokenized[i])):
            for k in range(len(conversations_tokenized[i][j])):
                tokens.append(conversations_tokenized[i][j][k])
        input_ids.append(tokens)

    # Adding Padding
    longest_len = {'train':400, 'validate':400, 'test':400}
    longest = 400

    pad_id = tokenizer.convert_tokens_to_ids('<pad>')
    for i in range(len(input_ids)):
        for j in range(longest-len(input_ids[i])):
            input_ids[i].append(pad_id)
            lm_labels[i].append(-100)
            token_type_ids[i].append(pad_id)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    lm_labels = torch.tensor(lm_labels, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)        

    return input_ids, lm_labels, token_type_ids

w_input_ids_train, w_lm_labels_train, w_token_type_ids_train = prepare_data('train', 'woz')
d_input_ids_train, d_lm_labels_train, d_token_type_ids_train = prepare_data('train', 'dstc2')
w_input_ids_valid, w_lm_labels_valid, w_token_type_ids_valid = prepare_data('validate', 'woz')
d_input_ids_valid, d_lm_labels_valid, d_token_type_ids_valid = prepare_data('validate', 'dstc2')
w_input_ids_test, w_lm_labels_test, w_token_type_ids_test = prepare_data('test', 'woz')
d_input_ids_test, d_lm_labels_test, d_token_type_ids_test = prepare_data('test', 'dstc2')

input_ids_train = torch.cat((w_input_ids_train, d_input_ids_train), 0)
lm_labels_train = torch.cat((w_lm_labels_train, d_lm_labels_train), 0)
token_type_ids_train = torch.cat((w_token_type_ids_train, d_token_type_ids_train), 0)

input_ids_valid = torch.cat((w_input_ids_valid, d_input_ids_valid), 0)
lm_labels_valid = torch.cat((w_lm_labels_valid, d_lm_labels_valid), 0)
token_type_ids_valid = torch.cat((w_token_type_ids_valid, d_token_type_ids_valid), 0)

input_ids_test = torch.cat((w_input_ids_test, d_input_ids_test), 0)
lm_labels_test = torch.cat((w_lm_labels_test, d_lm_labels_test), 0)
token_type_ids_test = torch.cat((w_token_type_ids_test, d_token_type_ids_test), 0)

# Creating custom training dataset
class TrainDataset(Dataset):
    def __init__(self, input_ids_train, lm_labels_train, token_type_ids_train):
        self.input_ids = input_ids_train
        self.lm_labels = lm_labels_train
        self.token_type_ids = token_type_ids_train
        self.n_samples = self.input_ids.shape[0]
        
    def __getitem__(self, index):
        return self.input_ids[index], self.lm_labels[index], self.token_type_ids[index]
    
    def __len__(self):
        return self.n_samples
    
# Creating custom validation dataset
class ValDataset(Dataset):
    def __init__(self, input_ids_valid, lm_labels_valid, token_type_ids_valid):
        self.input_ids = input_ids_valid
        self.lm_labels = lm_labels_valid
        self.token_type_ids = token_type_ids_valid
        self.n_samples = self.input_ids.shape[0]
        
    def __getitem__(self, index):
        return self.input_ids[index], self.lm_labels[index], self.token_type_ids[index]
    
    def __len__(self):
        return self.n_samples
    
# Creating custom testing dataset
class TestDataset(Dataset):
    def __init__(self, input_ids_test, lm_labels_test, token_type_ids_test):
        self.input_ids = input_ids_test
        self.lm_labels = lm_labels_test
        self.token_type_ids = token_type_ids_test
        self.n_samples = self.input_ids.shape[0]
        
    def __getitem__(self, index):
        return self.input_ids[index], self.lm_labels[index], self.token_type_ids[index]
    
    def __len__(self):
        return self.n_samples
    
train_dataset = TrainDataset(input_ids_train, lm_labels_train, token_type_ids_train)
val_dataset = ValDataset(input_ids_valid, lm_labels_valid, token_type_ids_valid)
test_dataset = TestDataset(input_ids_test, lm_labels_test, token_type_ids_test)
batch_size = 8

# Implementing train loader to split the data into batches
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True, # data reshuffled at every epoch
                          num_workers=0) # Use several subprocesses to load the data

# Implementing train loader to split the data into batches
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          shuffle=True, # data reshuffled at every epoch
                          num_workers=0) # Use several subprocesses to load the data

# Implementing train loader to split the data into batches
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True, # data reshuffled at every epoch
                          num_workers=0) # Use several subprocesses to load the data

val_samples =  len(val_loader)
n_samples = len(train_loader)
n_iterations = math.ceil(n_samples/batch_size)

print(input_ids_train.shape)
# longest = 400
print(input_ids_valid.shape)
# longest = 400
print(input_ids_test.shape)
# longest = 400

optimizer = AdamW(model.parameters(), lr=6.25e-5, correct_bias=True)
EPOCHS = 50

# Implementing checkpoints
def save_checkpoint_best(epoch, model):
    print("Saving best model")
    PATH = "/content/drive/My Drive/Capstone/conversational_models/woz_baseline/best_model_"+str(epoch)+".pt"
    torch.save(model.state_dict(), PATH)

def save_checkpoint(epoch, model, optimizer):  # Saving model in a way so we can load and start training again
    PATH = "/content/drive/My Drive/Capstone/conversational_models/woz_baseline/model_"+str(epoch)+".pt"
    print("Saving model")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH)
    
tr_loss_log = []
val_loss_log = []

example = iter(train_loader)
input_ids, lm_labels, token_type_ids = example.next()
writer.add_graph(model, input_ids.to(device))
writer.close()

# Training Loop
def train_model():
    
    least_val_loss = math.inf
    
    for epoch in range(EPOCHS):
        
        beg_time = time.time() #To calculate time taken for each epoch
        train_loss = 0.0
        val_loss = 0.0
        
        for i, (input_ids, lm_labels, token_type_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            lm_labels = lm_labels.to(device)
            token_type_ids = token_type_ids.to(device)
            # Forward pass
            loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=lm_labels)
            loss.backward()
            # Update gradients
            optimizer.step()
            # Get training loss
            train_loss += loss.item()
        tr_loss_log.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            for i, (input_ids, lm_labels, token_type_ids) in enumerate(val_loader):
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                lm_labels = lm_labels.to(device)
                token_type_ids = token_type_ids.to(device)
                # Forward pass
                loss, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=lm_labels)
                val_loss += loss.item()
            val_loss_log.append(val_loss)
        model.train()
      
        # Saving checkpoints
        #save_checkpoint(epoch+1, model, optimizer)
        if(val_loss < least_val_loss):
            save_checkpoint_best(epoch+1, model)
            least_val_loss = val_loss
          
        end_time = time.time()
        print('Epoch: {:.0f}/{:.0f}, Time: {:.0f}m {:.0f}s, Train_Loss: {:.4f}, Val_Loss: {:.4f}'.format(
            epoch+1, EPOCHS, (end_time-beg_time)//60, (end_time-beg_time)%60, train_loss, val_loss))
        writer.add_scalar('Training_loss', train_loss, (epoch+1))
        writer.add_scalar('Validation_loss', val_loss, (epoch+1))
        writer.close()
        

train_model()