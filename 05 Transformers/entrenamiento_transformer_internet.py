# Importing libraries

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# HuggingFace libraries
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Pathlib
from pathlib import Path

# typing
from typing import Any

# Library for progress bars in loops
from tqdm import tqdm

from transformer_internet import get_model, MI_TRANSFORMER, MI_ENCODER, MI_DECODER, MI_PROJECTION

SUBSET = True
SUBSET_ONE_SAMPLE = False
PERCENT_SUBSET = 0.01
LEN_SUBSET_ONE_SAMPLE = 1

if SUBSET:
    if MI_ENCODER and MI_DECODER and MI_PROJECTION:
        BS = 40
    elif MI_ENCODER and MI_DECODER:
        BS = 32
    elif MI_ENCODER and MI_PROJECTION:
        BS = 32
    elif MI_DECODER and MI_PROJECTION:
        BS = 32
    elif MI_ENCODER:
        BS = 24
    elif MI_DECODER:
        BS = 24
    elif MI_PROJECTION:
        BS = 24
    else:
        BS = 24
    if SUBSET_ONE_SAMPLE:
        BS = 1
else:
    BS = 16

SOURCE_LANGUAGE = 'en'
TARGET_LANGUAGE = 'es'
EPOCHS = 200
if MI_ENCODER and MI_DECODER and MI_PROJECTION:
    LR = 10**-6
else:
    LR = 10**-4
MAX_SECUENCE_LEN = 350
DIM_EMBEDDING = 512

# Defining Tokenizer
def build_tokenizer(config, ds, lang):

    # Crating a file path for the tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    print(f'Tokenizer path: {tokenizer_path}')

    # Checking if Tokenizer already exists
    if not Path.exists(tokenizer_path):

        # If it doesn't exist, we create a new one
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]')) # Initializing a new world-level tokenizer
        tokenizer.pre_tokenizer = Whitespace() # We will split the text into tokens based on whitespace

        # Creating a trainer for the new tokenizer
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]",
                                                     "[SOS]", "[EOS]"], min_frequency = 2) # Defining Word Level strategy and special tokens

        # Training new tokenizer on sentences from the dataset and language specified
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        # tokenizer.save(str(tokenizer_path)) # Saving trained tokenizer to the file path specified at the beginning of the function
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # If the tokenizer already exist, we load it
    return tokenizer # Returns the loaded tokenizer or the trained tokenizer

# Iterating through dataset to extract the original sentence and its translation
def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair['translation'][lang]

def get_ds(config):

    # Loading the train portion of the OpusBooks dataset.
    # The Language pairs will be defined in the 'config' dictionary we will build later
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    if SUBSET:
      if SUBSET_ONE_SAMPLE:
            ds_raw = ds_raw.select(range(LEN_SUBSET_ONE_SAMPLE))
      else:
            ds_raw = ds_raw.select(range(int(PERCENT_SUBSET*len(ds_raw))))
    print(f'Number of examples in the dataset: {len(ds_raw)}')

    # Building or loading tokenizer for both the source and target languages
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Splitting the dataset for training and validation
    train_ds_size = int(0.9 * len(ds_raw)) # 90% for training
    val_ds_size = len(ds_raw) - train_ds_size # 10% for validation
    if SUBSET_ONE_SAMPLE:
        train_ds_raw = ds_raw.select(range(LEN_SUBSET_ONE_SAMPLE))
        val_ds_raw = train_ds_raw
    else:
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # Randomly splitting the dataset

    # Processing data with the BilingualDataset class, which we will define below
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
    max_len_src = 0
    max_len_tgt = 0
    for pair in ds_raw:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Creating dataloaders for the training and validadion sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True) # Batch size will be defined in the config dictionary
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt # Returning the DataLoader objects and tokenizers

def casual_mask(size):
        # Creating a square matrix of dimensions 'size x size' filled with ones
        mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
        return mask == 0

class BilingualDataset(Dataset):

    # This takes in the dataset contaning sentence pairs, the tokenizers for target and source languages, and the strings of source and target languages
    # 'seq_len' defines the sequence length for both languages
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Defining special tokens by using the target language tokenizer
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    # Total number of instances in the dataset (some pairs are larger than others)
    def __len__(self):
        return len(self.ds)

    # Using the index to retrive source and target texts
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenizing source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncating or padding the tokenized texts to the defined 'seq_len'
        enc_input_tokens = enc_input_tokens[:self.seq_len-2]
        dec_input_tokens = dec_input_tokens[:self.seq_len-1]

        # Computing how many padding tokens need to be added to the tokenized texts
        # Source tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # Subtracting the two '[EOS]' and '[SOS]' special tokens
        # Target tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Subtracting the '[SOS]' special token

        # If the texts exceed the 'seq_len' allowed, it will raise an error. This means that one of the sentences in the pair is too long to be processed
        # given the current sequence length limit (this will be defined in the config dictionary below)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Building the encoder input tensor by combining several elements
        encoder_input = torch.cat(
            [
            self.sos_token, # inserting the '[SOS]' token
            torch.tensor(enc_input_tokens, dtype = torch.int64), # Inserting the tokenized source text
            self.eos_token, # Inserting the '[EOS]' token
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64) # Addind padding tokens
            ]
        )

        # Building the decoder input tensor by combining several elements
        decoder_input = torch.cat(
            [
                self.sos_token, # inserting the '[SOS]' token
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Inserting the tokenized target text
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Addind padding tokens
            ]

        )

        # Creating a label tensor, the expected output for training the model
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Inserting the tokenized target text
                self.eos_token, # Inserting the '[EOS]' token
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Adding padding tokens

            ]
        )

        # Ensuring that the length of each tensor above is equal to the defined 'seq_len'
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'input_to_encoder_tokeniced': encoder_input,
            'input_to_decoder_tokeniced': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'target_to_decoder_tokeniced': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
# Define function to obtain the most probable next token
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    debug = False
    # Retrieving the indices from the start and end of sequences of the target tokens
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Computing the output of the encoder for the source sequence
    if MI_ENCODER:
        encoder_output = model.encode(source, source_mask)
    else:
        encoder_output = model.encode(source, source_mask)
    if debug: print('*'*80)
    if debug: print(f"encoder_output shape: {encoder_output.shape} (1, max_seq_len, dim_embedding) = (1, 350, 512)") # (1, max_seq_len, dim_embedding) = (1, 350, 512)
    # Initializing the decoder input with the Start of Sentence token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    # Looping until the 'max_len', maximum length, is reached
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Building a mask for the decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculating the output of the decoder
        if MI_DECODER:
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        else:
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        if debug: print(f"\tout shape: {out.shape} (1, seq_len, dim_embedding) = (1, seq_len, 512)") # (1, seq_len, dim_embedding) = (1, seq_len, 512)

        # Applying the projection layer to get the probabilities for the next token
        if MI_PROJECTION:
            prob = model.project(out[:, -1])
        else:
            prob = model.project(out[:, -1])
        if debug: print(f"\tprob shape: {prob.shape} (1, target_vocab_size)") # (1, target_vocab_size)
        if debug: print('*'*80)

        # Selecting token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1). type_as(source).fill_(next_word.item()).to(device)], dim=1)

        # If the next token is an End of Sentence token, we finish the loop
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # Sequence of tokens generated by the decoder

# Define settings for building and training the transformer model
def get_config():
    return{
        'batch_size':BS,
        'num_epochs': EPOCHS,
        'lr': LR,
        'seq_len': MAX_SECUENCE_LEN,
        'd_model': DIM_EMBEDDING,
        'lang_src': SOURCE_LANGUAGE,
        'lang_tgt': TARGET_LANGUAGE,
        'tokenizer_file': 'tokenizer_{0}.json',
    }

def train_loop(model, loss_fn, optimizer, tokenizer_tgt, device, global_step, batch_iterator):
    # For each batch...
    for batch in batch_iterator:
        model.train() # Train the model

        # Loading input data and masks onto the GPU
        input_to_encoder_tokeniced = batch['input_to_encoder_tokeniced'].to(device)
        input_to_decoder_tokeniced = batch['input_to_decoder_tokeniced'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)

        # Running tensors through the Transformer
        if MI_TRANSFORMER:
            encoder_output = model.encode(input_to_encoder_tokeniced)
            decoder_output = model.decode(input_to_decoder_tokeniced, encoder_output, decoder_mask)
            proj_output = model.linear_and_softmax(decoder_output)
        else:
            encoder_output = model.encode(input_to_encoder_tokeniced, encoder_mask)
            decoder_output = model.decode(encoder_output=encoder_output, src_mask=encoder_mask, tgt=input_to_decoder_tokeniced, tgt_mask=decoder_mask)
            proj_output = model.project(decoder_output)
        
        # Loading the target labels onto the GPU
        label = batch['target_to_decoder_tokeniced'].to(device)

        # Computing loss between model's output and true labels
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

        # Updating progress bar
        batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

        # writer.add_scalar('train loss', loss.item(), global_step)
        # writer.flush()

        # Performing backpropagation
        loss.backward()

        # Updating parameters based on the gradients
        optimizer.step()

        # Clearing the gradients to prepare for the next batch
        optimizer.zero_grad()

        global_step += 1 # Updating global step count

    return global_step, model, optimizer

# Defining function to evaluate the model on the validation dataset
# num_examples = 2, two examples per run
def validation_loop(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    model.eval() # Setting model to evaluation mode
    count = 0 # Initializing counter to keep track of how many examples have been processed

    console_width = 80 # Fixed witdh for printed messages

    # Creating evaluation loop
    with torch.no_grad(): # Ensuring that no gradients are computed during this process
        for batch in validation_ds:
            count += 1
            input_to_encoder_tokeniced = batch['input_to_encoder_tokeniced'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Ensuring that the batch_size of the validation set is 1
            assert input_to_encoder_tokeniced.size(0) ==  1, 'Batch size must be 1 for validation.'

            # Applying the 'greedy_decode' function to get the model's output for the source text of the input batch
            model_out = greedy_decode(model, input_to_encoder_tokeniced, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Retrieving source and target texts from the batch
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0] # True translation
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # Decoded, human-readable model output

            # Printing results
            print_msg('-'*console_width)
            print_msg(f'SOURCE EXAMPLE {count}: {source_text}')
            print_msg(f'TARGET EXAMPLE {count}: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            # print(f"model_out shape: {model_out.shape}, model_out: {model_out}")

            # After two examples, we break the loop
            if count == num_examples:
                break

def debug_one_sample_of_dataloder(train_dataloader):
    sample_batch = next(iter(train_dataloader))
    print('*'*80)
    print(f"Input to encoder tokeniced shape: {sample_batch['input_to_encoder_tokeniced'].shape} (batch_size, seq_len)")  # (batch_size, seq_len)
    print(f"Input to encoder tokeniced dtype: {sample_batch['input_to_encoder_tokeniced'].dtype}")
    print(f"Input to decoder tokeniced shape: {sample_batch['input_to_decoder_tokeniced'].shape} (batch_size, seq_len)")  # (batch_size, seq_len)
    print(f"Input to decoder tokeniced dtype: {sample_batch['input_to_decoder_tokeniced'].dtype}")
    print(f"Encoder mask shape: {sample_batch['encoder_mask'].shape} (batch_size, 1, 1, seq_len)")                        # (batch_size, 1, 1, seq_len)
    print(f"Encoder mask dtype: {sample_batch['encoder_mask'].dtype}")
    print(f"Decoder mask shape: {sample_batch['decoder_mask'].shape} (batch_size, 1, seq_len, seq_len)")                  # (batch_size, 1, seq_len, seq_len)
    print(f"Decoder mask dtype: {sample_batch['decoder_mask'].dtype}")
    print(f"Target to decoder tokeniced shape: {sample_batch['target_to_decoder_tokeniced'].shape} (batch_size, seq_len)")# (batch_size, seq_len)
    print(f"Target to decoder tokeniced dtype: {sample_batch['target_to_decoder_tokeniced'].dtype}")
    print(f"Source text example:\n{sample_batch['src_text'][0]}")
    print(f"Target text example:\n{sample_batch['tgt_text'][0]}")
    print('*'*80)

def train_model(config):
    # Setting up device to run on GPU to train faster
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Debugging one sample of the dataloader
    debug_one_sample_of_dataloder(train_dataloader)

    # Initializing model on the GPU using the 'get_model' function
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    print(f'Source vocabulary size: {src_vocab_size}')
    print(f'Target vocabulary size: {tgt_vocab_size}')
    dim_embedding = config['d_model']
    print(f'Embedding dimension: {dim_embedding}')
    src_seq_len = config['seq_len']
    print(f'Source max sequence length: {src_seq_len}')
    tgt_seq_len = config['seq_len']
    print(f'Target max sequence length: {tgt_seq_len}')
    Nx = 6 #Nx,
    h = 8 #config['h']
    dropout = 0.1 #config['dropout']
    d_ff = 2048 #config['d_ff']
    print(f"Nx: {Nx}, h: {h}, dropout: {dropout}, d_ff: {d_ff}")
    print(f"SOS token id: {tokenizer_tgt.token_to_id('[SOS]')}, EOS token id: {tokenizer_tgt.token_to_id('[EOS]')}, PAD token id: {tokenizer_tgt.token_to_id('[PAD]')}, UNK token id: {tokenizer_tgt.token_to_id('[UNK]')}")
    print('*'*80)
    model = get_model(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, dim_embedding, Nx, h, dropout, d_ff).to(device)

    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0

    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    # Initializing training loop

    # Iterating over each epoch from the 'initial_epoch' variable up to
    # the number of epochs informed in the config
    for epoch in range(initial_epoch, config['num_epochs']):

        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')

        # Running the training loop for one epoch
        global_step, model, optimizer = train_loop(model, loss_fn, optimizer, tokenizer_tgt, device, global_step, batch_iterator)

        # We run the 'validation_loop' function at the end of each epoch
        # to evaluate model performance
        validation_loop(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

config = get_config() # Retrieving config settings

train_model(config) # Training model with the config arguments