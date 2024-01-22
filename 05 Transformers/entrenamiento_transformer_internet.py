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

# Importing library of warnings
import warnings

# Transformer library
from transformer_internet import *
from transformer import MiTransformer, MiEncoder, MiDecoder, Linear_and_softmax, MiEmbedding, MiPositionalEncoding

# Validation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

SUBSET = False
SUBSET_ONE_SAMPLE = False
PERCENT_SUBSET = 0.1
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
        BS = 16
    if SUBSET_ONE_SAMPLE:
        BS = 1
else:
    BS = 20
print(f"MI_EMBEDDINGS: {MI_EMBEDDINGS}")
print(f"MI_POSITIONAL_ENCODING: {MI_POSITIONAL_ENCODING}")
print(f"MI_ENCODER: {MI_ENCODER}")
print(f"MI_DECODER: {MI_DECODER}")
print(f"MI_PROJECTION: {MI_PROJECTION}")
print(f"BS: {BS}")

SOURCE_LANGUAGE = 'en'
TARGET_LANGUAGE_ES = 'es'
TARGET_LANGUAGE_IT = 'it'
TARGET_LANGUAGE = TARGET_LANGUAGE_IT
EPOCHS = 200
LR = 10**-4
MAX_SECUENCE_LEN = 350
DIM_EMBEDDING = 512

def build_tokenizer(config, ds, lang):
    
    # Crating a file path for the tokenizer 
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
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
    percen_train = 0.99 # Percentage of the dataset that will be used for training
    train_ds_size = int(percen_train * len(ds_raw)) # 90% for training
    val_ds_size = len(ds_raw) - train_ds_size # 10% for validation
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # Randomly splitting the dataset
                                    
    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
    max_len_src = 0
    max_len_tgt = 0
    for pair in ds_raw:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    max_sec_len = max(max_len_src, max_len_tgt)
    max_sec_len = max_sec_len + 2 # Adding 2 to account for the '[SOS]' and '[EOS]' tokens
    config['seq_len'] = max_sec_len # Adding the maximum length to the config dictionary
        
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    print(f'Maximum sequence length: {max_sec_len}')
                                    
    # Processing data with the BilingualDataset class, which we will define below
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], max_sec_len)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], max_sec_len)
    
    # Creating dataloaders for the training and validadion sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True) # Batch size will be defined in the config dictionary
    val_dataloader = DataLoader(val_ds, batch_size = config['batch_size'], shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, max_sec_len # Returning the DataLoader objects and tokenizers

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
            'encoder_input': encoder_input,
            'decoder_input': decoder_input, 
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), 
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    # Creating Embedding layers
    if MI_EMBEDDINGS:
        src_embed = MiEmbedding(src_vocab_size, d_model)
        tgt_embed = MiEmbedding(tgt_vocab_size, d_model)
    else:
        src_embed = InputEmbeddings(d_model, src_vocab_size) # Source language (Source Vocabulary to 512-dimensional vectors)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size) # Target language (Target Vocabulary to 512-dimensional vectors)
    
    # Creating Positional Encoding layers
    if MI_POSITIONAL_ENCODING:
        src_pos = MiPositionalEncoding(max_sequence_len=src_seq_len, embedding_model_dim=d_model)
        tgt_pos = MiPositionalEncoding(max_sequence_len=tgt_seq_len, embedding_model_dim=d_model)
    else:
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # Positional encoding for the source language embeddings
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # Positional encoding for the target language embeddings
    
    # Creating EncoderBlocks
    if not MI_ENCODER:
        encoder_blocks = [] # Initial list of empty EncoderBlocks
        for _ in range(N): # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
            
            # Combine layers into an EncoderBlock
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block) # Appending EncoderBlock to the list of EncoderBlocks
        
    # Creating DecoderBlocks
    if not MI_DECODER:
        decoder_blocks = [] # Initial list of empty DecoderBlocks
        for _ in range(N): # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Cross-Attention
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
            
            # Combining layers into a DecoderBlock
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block) # Appending DecoderBlock to the list of DecoderBlocks
        
    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    if MI_ENCODER:
        encoder = MiEncoder(h, d_model, N, dropout)
    else:
        encoder = Encoder(nn.ModuleList(encoder_blocks))
    if MI_DECODER:
        decoder = MiDecoder(h, d_model, N, dropout)
    else:
        decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Creating projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # Map the output of Decoder to the Target Vocabulary Space
    
    # Creating the transformer by combining everything above
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer # Assembled and initialized Transformer. Ready to be trained and validated!
    
def get_model(config, vocab_src_len, vocab_tgt_len, max_sec_len):
    
    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, max_sec_len, max_sec_len, config['d_model'])
    return model

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, bs=1):
    # Retrieving the indices from the start and end of sequences of the target tokens
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')    # Start of Sentence token index (2)
    # eos_idx = tokenizer_tgt.token_to_id('[EOS]')    # End of Sentence token index (3)

    # Computing the output of the encoder for the source sequence
    if MI_ENCODER:
        encoder_output = model.encode(source, source_mask)
    else:
        encoder_output = model.encode(source, source_mask)
    
    # Initializing the decoder input with the Start of Sentence token
    decoder_input = torch.empty(bs,1).fill_(sos_idx).type_as(source).to(device)
    
    # Looping until the 'max_len', maximum length, is reached
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        # Building a mask for the decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculating the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Applying the projection layer to get the probabilities for the next token
        prob = model.project(out[:, -1])

        # Selecting token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        # decoder_input = torch.cat([decoder_input, torch.empty(1,1). type_as(source).fill_(next_word.item()).to(device)], dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
    
    if len(decoder_input.shape) == 1:
        decoder_input = decoder_input.unsqueeze(0)
    elif len(decoder_input.shape) == 3:
        decoder_input = decoder_input.squeeze(0)

    return decoder_input # Sequence of tokens generated by the decoder

def get_config():
    return{
        'batch_size': BS,
        'num_epochs': EPOCHS,
        'lr': LR,
        'seq_len': MAX_SECUENCE_LEN,
        'd_model': DIM_EMBEDDING,
        'lang_src': SOURCE_LANGUAGE,
        'lang_tgt': TARGET_LANGUAGE,
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder'] # Extracting model folder from the config
    model_basename = config['model_basename'] # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt" # Building filename
    return str(Path('.')/ model_folder/ model_filename) # Combining current directory, the model folder, and the model filename

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

        # Performing backpropagation
        loss.backward()

        # Updating parameters based on the gradients
        optimizer.step()

        # Clearing the gradients to prepare for the next batch
        optimizer.zero_grad()

        global_step += 1 # Updating global step count

    return global_step, model, optimizer

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    model.eval() # Setting model to evaluation mode

    # Calculating the number of batches in the validation dataset
    dataset_size = len(validation_ds.dataset)  # Tamaño total del conjunto de datos
    batch_size = validation_ds.batch_size      # Tamaño del batch
    drop_last = validation_ds.drop_last        # Configuración de drop_last
    num_batches = len(validation_ds)           # Número total de batches

    # Calculating the total number of samples in the validation dataset
    total_samples = batch_size * (num_batches - 1) + min(batch_size, dataset_size % batch_size)

    # If drop_last is False and the dataset size is not divisible by the batch size, we need to add one more batch
    if drop_last and dataset_size % batch_size != 0:
        total_samples -= dataset_size % batch_size

    # Initializing progress bar
    progress_bar = tqdm(range(total_samples), desc = 'Processing validation examples') # Initializing progress bar

    # Initializing lists to store scores
    bleu_scores = []
    meteor_scores = []
    
    # Creating evaluation loop
    with torch.no_grad(): # Ensuring that no gradients are computed during this process
        for batch in validation_ds:
            # Loading input data and masks onto the GPU
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Applying the 'greedy_decode' function to get the model's output for the source text of the input batch
            num_samples = len(batch['src_text'])
            model_out_bs = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, bs=num_samples)

            # Get metrics for every example in the batch
            for i in range(num_samples):
                source_text = batch['src_text'][i]
                target_text = batch['tgt_text'][i]
                model_out_i = model_out_bs[i]
                model_out_text = tokenizer_tgt.decode(model_out_i.detach().cpu().numpy())

                # Calculating metrics
                references = [target_text.split()]
                hypothesis = model_out_text.split()
                bleu_score = sentence_bleu(references, hypothesis)
                meteor_score_value = meteor_score(references, hypothesis)
            
                # Appending scores to lists
                bleu_scores.append(bleu_score)
                meteor_scores.append(meteor_score_value)

                # Calculating mean scores            
                mean_bleu_score = sum(bleu_scores)/len(bleu_scores) # Calculating mean BLEU score
                mean_meteor_score = sum(meteor_scores)/len(meteor_scores) # Calculating mean METEOR score

                # Updating progress bar and printing bleu and meteor scores
                progress_bar.update(1)
                progress_bar.set_postfix({'BLEU': f'{mean_bleu_score:.9f}', 'METEOR': f'{mean_meteor_score:.9f}'})

    # Printing results
    console_width = 80 # Fixed witdh for printed messages
    print('-'*console_width)
    print(f'SOURCE: {source_text}')
    print(f'TARGET: {target_text}')
    print(f'PREDICTED: {model_out_text}')
    print('-'*console_width)

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
    
    # Creating model directory to store weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, max_sec_len = get_ds(config)
    
    # Initializing model on the GPU using the 'get_model' function
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), max_sec_len).to(device)
    
    # Tensorboard
    # writer = SummaryWriter(config['experiment_name'])
    
    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    
    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0
    
    # Checking if there is a pre-trained model to load
    # If true, loads it
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) # Loading model
        
        # Sets epoch to the saved in the state plus one, to resume from where it stopped
        initial_epoch = state['epoch'] + 1
        # Loading the optimizer state from the saved model
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # Loading the global step state from the saved model
        global_step = state['global_step']
        
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
        print()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        
        # For each batch...
        for batch in batch_iterator:
            model.train() # Train the model
            
            # Loading input data and masks onto the GPU
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Running tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            # Loading the target labels onto the GPU
            label = batch['label'].to(device)
            
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
            
        # We run the 'run_validation' function at the end of each epoch
        # to evaluate model performance
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))
         
        # Saving model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # Writting current model state to the 'model_filename'
        # torch.save({
        #     'epoch': epoch, # Current epoch
        #     'model_state_dict': model.state_dict(),# Current model state
        #     'optimizer_state_dict': optimizer.state_dict(), # Current optimizer state
        #     'global_step': global_step # Current global step 
        # }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore') # Filtering warnings
    config = get_config() # Retrieving config settings
    train_model(config) # Training model with the config arguments