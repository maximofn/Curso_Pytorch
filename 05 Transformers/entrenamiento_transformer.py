import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import tiktoken
import numpy as np
import time
import os

from transformer import Transformer

from sacrebleu import corpus_bleu
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge 

nltk.download('wordnet')
rouge = Rouge()

train_inputs_path = "data/opus100_croped_10/train_inputs.pt"
train_labels_path = "data/opus100_croped_10/train_labels.pt"
train_inputs = torch.load(train_inputs_path)
train_labels = torch.load(train_labels_path)
test_inputs_path = "data/opus100_croped_10/test_inputs.pt"
test_labels_path = "data/opus100_croped_10/test_labels.pt"
test_inputs = torch.load(test_inputs_path)
test_labels = torch.load(test_labels_path)
validation_inputs_path = "data/opus100_croped_10/validation_inputs.pt"
validation_labels_path = "data/opus100_croped_10/validation_labels.pt"

validation_inputs = torch.load(validation_inputs_path)
validation_labels = torch.load(validation_labels_path)

EPOCH0 = 0
STEP0 = 0
LR = 1e7
UPDATE_LR = False
SUBSET = True
LEN_SUBSET_TRAIN = 1000
if LEN_SUBSET_TRAIN == 1:
    test_inputs = train_inputs
    test_labels = train_labels
    validation_inputs = train_inputs
    validation_labels = train_labels
LEN_SUBSET_VALIDATION = min(max(int(LEN_SUBSET_TRAIN/10), 1), len(validation_inputs))
LEN_SUBSET_TEST = min(max(int(LEN_SUBSET_TRAIN/10), 1), len(test_inputs))
MODEL_PATH = f"model"
EPOCHS = 100000
GPUS = 1
GPU_NUMBER = 0
if GPUS > 1:
    BS = 350
else:
    BS = 350
DIMENSION_EMBEDDING = 32 #512
NUMBER_OF_HEADS = 8 # 8
NUMBER_OF_TRANSFORMER_BLOCKS = 4 # 6

if os.path.exists(MODEL_PATH):
    files = os.listdir(MODEL_PATH)
    for file in files:
        if "transformer" in file:
            name = file.split(".")[0]
            STEP0 = int(name.split("_")[-1])
            EPOCH0 = int(name.split("_")[-2])

encoder = tiktoken.get_encoding("cl100k_base")
start_token = encoder.encode(chr(1))
end_token = encoder.encode(chr(2))
padding_token = encoder.encode(chr(3))

max_secuence_length = 10+2 #104
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
validation_dataset = TensorDataset(validation_inputs, validation_labels)

if SUBSET:
    def subsample_dataset(dataset, new_size):
        indices = np.random.choice(len(dataset), new_size, replace=False)
        indices = indices.tolist()  # Convert numpy.int64 indices to native int
        return torch.utils.data.Subset(dataset, indices)
    train_dataset = subsample_dataset(train_dataset, LEN_SUBSET_TRAIN)
    validation_dataset = subsample_dataset(validation_dataset, LEN_SUBSET_VALIDATION)
    test_dataset = subsample_dataset(test_dataset, LEN_SUBSET_TEST)
    print(f"SUBSET: train: {len(train_dataset)}, validation: {len(validation_dataset)}, test: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BS, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

def prepare_source_sentence(sentence, start_token, end_token, pad_token, max_length, device):
    sentence = encoder.encode(sentence)
    sentence = start_token + sentence + end_token
    if len(sentence) < max_length:
        sentence = sentence + pad_token * (max_length - len(sentence))
    else:
        sentence = sentence[:max_length]
    sentence = torch.tensor([sentence]).to(device)
    return sentence

def prepare_target_sentence(sentence, start_token, pad_token, max_length, device):
    sentence = encoder.encode(sentence)
    sentence = start_token + sentence + end_token
    if len(sentence) < max_length:
        sentence = sentence + pad_token * (max_length - len(sentence))
    else:
        sentence = sentence[:max_length]
    sentence = torch.tensor([sentence]).to(device)
    return sentence

vocab_size = encoder.n_vocab
prob_dropout = 0.1
transformer = Transformer(vocab_size, DIMENSION_EMBEDDING, max_secuence_length, NUMBER_OF_HEADS, NUMBER_OF_TRANSFORMER_BLOCKS, prob_dropout)

if torch.cuda.device_count() > 1 and GPUS > 1:
    number_gpus = torch.cuda.device_count()
    print(f"Let's use {number_gpus} GPUs!")
    transformer = nn.DataParallel(transformer)

    def create_mask(sequence_len):
        mask = torch.tril(torch.ones((number_gpus*sequence_len, sequence_len)))
        return mask
else:
    def create_mask(sequence_len):
        mask = torch.tril(torch.ones((sequence_len, sequence_len)))
        return mask

mask = create_mask(max_secuence_length)

def get_target_sentence(source, target, mask, model, device, end_token, max_len):
    model = model.to(device)
    # model.eval()
    source = source.to(device)
    target = target.to(device)
    mask = mask.to(device)
    end_token = torch.tensor(end_token)
    output_sentence = target.clone()

    for i in range(max_len-2):
        with torch.no_grad():
            output = model(source, target, mask)
            next_token = output[0, i+1].argmax().item()
            output_sentence[0, i+1] = next_token
            if next_token == end_token:
                break
    output_sentence[0, max_len-1] = end_token

    return output_sentence

def decode_sentence(sentence, decoder, end_token):
    decoded = ""
    if isinstance(end_token, list):
        end_token = end_token[0]
    if isinstance(sentence, torch.Tensor):
        sentence = sentence.cpu().numpy()
    if end_token in sentence:
        position_end_token = np.where(sentence == end_token)[0]
        if len(position_end_token) > 1:
            position_end_token = position_end_token[0]
        position_end_token = int(position_end_token)
        sentence = sentence[:position_end_token+1]
    sentence = sentence[1:-1]   # Remove start and end token
    try:
        decoded = decoder(sentence)
    except:
        Warning(f"Error decoding sentence: {sentence}")
        decoded = ""
    return decoded

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            mask = (target == self.ignore_index).unsqueeze(1).expand_as(true_dist)
            if mask.any():
                true_dist[mask] = 0
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# loss_function = LabelSmoothingLoss(classes=vocab_size, smoothing=0.1, ignore_index=padding_token[0])
# loss_function = nn.CrossEntropyLoss(ignore_index=padding_token[0], label_smoothing=0.1)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

class Step():
    def __init__(self):
        self.step = 0
    
    def set_step(self, st):
        self.step = st
    
    def get_step(self):
        return int(self.step)

class LearningRate():
    def __init__(self):
        self.lr = 0
    
    def set_lr(self, l_r_):
        self.lr = l_r_
    
    def get_lr(self):
        return self.lr

actual_step = Step()
actual_lr = LearningRate()

def calculate_lr(step_num, dim_embedding_model=512, warmup_steps=4000):
    step_num += 1e-7 # Avoid division by zero
    step_num += STEP0
    actual_step.set_step(step_num)
    step_num_exp = -0.7
    warmup_steps_exp = -2.0
    dim_embedding_model_exp = -0.001
    lr = np.power(dim_embedding_model, dim_embedding_model_exp) * np.minimum(np.power(step_num, step_num_exp), step_num * np.power(warmup_steps, warmup_steps_exp))
    actual_lr.set_lr(lr)
    return lr

if UPDATE_LR:
    lr_lambda = lambda step: calculate_lr(step, dim_embedding_model=DIMENSION_EMBEDDING)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_loop(dataloader, model, loss_fn, optimizer, device, mask, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mean_loss = 0
    lr_list = []
    model.train()
    for batch, (src, trg) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)
        mask = mask.to(device)

        pred = model(src, trg, mask)
        loss = loss_fn(pred.view(-1, encoder.n_vocab), trg.view(-1))
        mean_loss += loss.item()

        # lr = optimizer.param_groups[0]['lr']
        if UPDATE_LR:
            lr = actual_lr.get_lr()
            lr_list.append(lr)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if UPDATE_LR:
            scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(src)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"loss: {loss:0.10f} [{current:>5d}/{size:>5d}] - epoch {epoch} - lr {current_lr:0.19f} - {(time.localtime().tm_year):02d}-{(time.localtime().tm_mon):02d}-{(time.localtime().tm_mday):02d}, {(time.localtime().tm_hour):02d}:{(time.localtime().tm_min):02d}:{(time.localtime().tm_sec):02d}")
    return mean_loss/num_batches, np.array(lr_list), model

if SUBSET and LEN_SUBSET_TRAIN < 100:
    sample = next(iter(train_dataloader))
    src, trg = sample
    encoded_src = src[0].cpu().numpy()
    decoded_src = decode_sentence(encoded_src, encoder.decode, end_token)
    print(f"English sentence: {decoded_src}")

def validation_loop(dataloader, model, loss_fn, device, mask, epoch):
    num_batches = len(dataloader)
    validation_loss = 0
    correct_predictions = 0
    total_predictions = 0
    # model.eval()

    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            mask = mask.to(device)

            pred = model(src, trg, mask)
            loss = loss_fn(pred.transpose(1, 2), trg)
            validation_loss += loss.item()

            predicted_ids = torch.argmax(pred, dim=2)
            for pred_sent, trg_sent in zip(predicted_ids, trg):
                if torch.equal(pred_sent, trg_sent):
                    correct_predictions += 1
            total_predictions += trg.size(0)

    validation_loss /= num_batches
    accuracy = correct_predictions / total_predictions
    pred_sent_decoded = decode_sentence(pred_sent, encoder.decode, end_token)
    trg_sent_decoded = decode_sentence(trg_sent, encoder.decode, end_token)
    bleu_score = corpus_bleu(pred_sent_decoded, trg_sent_decoded)
    pred_sent_tokenized = word_tokenize(pred_sent_decoded)
    trg_sent_tokenized = word_tokenize(trg_sent_decoded)
    meteor = meteor_score([pred_sent_tokenized], trg_sent_tokenized)
    rouge_score = rouge.get_scores(pred_sent_decoded, trg_sent_decoded, avg=True)
    print(f"Epoch {epoch} average validation loss: {validation_loss:0.8f}, best loss: {best_loss:0.8f}, accuracy: {(100*accuracy):>3.2f}% ({correct_predictions}/{total_predictions}), bleu score: {bleu_score.score:0.8f}, meteor: {meteor:0.8f}, rouge: {rouge_score['rouge-l']['f']:0.8f}")

    if SUBSET and LEN_SUBSET_TRAIN < 100:
        sentence_en = decoded_src
    else:
        sentence_en = "I have learned a lot from this course"
    encode_sentence_en = prepare_source_sentence(sentence_en, start_token, end_token, padding_token, max_secuence_length, device)
    sentence_es = ""
    encode_sentence_es = prepare_target_sentence(sentence_es, start_token, padding_token, max_secuence_length, device)
    encode_sentence_es = get_target_sentence(encode_sentence_en, encode_sentence_es, mask, model, device, end_token, max_secuence_length).squeeze(0).cpu().numpy()
    sentence_es = decode_sentence(encode_sentence_es, encoder.decode, end_token)
    print(f"Epoch {epoch}: English sentence:    {sentence_en}")
    print(f"Epoch {epoch}: Spanish translation: {sentence_es}")
    return validation_loss, accuracy

def elapsed_time(t0):
    t = time.time() - t0  # tiempo transcurrido en segundos

    # Convertir el tiempo a días, horas, minutos y segundos
    days, t = divmod(t, 86400)  # 86400 segundos en un día
    hours, t = divmod(t, 3600)  # 3600 segundos en una hora
    minutes, seconds = divmod(t, 60)  # 60 segundos en un minuto

    return int(days), int(hours), int(minutes), int(seconds)

def elapsed_seconds(days, hours, minutes, seconds):
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

if torch.cuda.device_count() > 1 and GPUS > 1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_NUMBER}")
        print(f"Using GPU {GPU_NUMBER}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
transformer = transformer.to(device)

if EPOCH0 != 0 or STEP0 != 0:
    if GPUS > 1:
        print(f"Loading model from {MODEL_PATH}/transformer_{EPOCH0}_{STEP0}.pth")
        transformer.load_state_dict(torch.load(f'{MODEL_PATH}/transformer_{EPOCH0}_{STEP0}.pth'))
        print(f"Model loaded from {MODEL_PATH}/transformer_{EPOCH0}_{STEP0}.pth")
    else:
        print(f"Loading model from {MODEL_PATH}/transformer_model_{EPOCH0}_{STEP0}.pth")
        weights = f"{MODEL_PATH}/transformer_model_{EPOCH0}_{STEP0}.pth"
        transformer = torch.load(weights, map_location=device)
        if isinstance(transformer, nn.DataParallel):
            print("Transforming model to nn.Module")
            transformer = transformer.module
        print(f"Model loaded from {MODEL_PATH}/transformer_{EPOCH0}_{STEP0}.pth")

    print(f"load best validation loss from {MODEL_PATH}/best_validation_loss_{EPOCH0}_{STEP0}.npy")
    best_loss = np.load(f'{MODEL_PATH}/validation_loss_{EPOCH0}_{STEP0}.npy')
    best_loss = best_loss[0]
    best_loss = float(best_loss)
    print(f"best validation loss: {best_loss}")
else:
    best_loss = 1000000

train_loss_list = []
train_lr_list = []
validation_loss_list = []
train_loss_list = np.array(train_loss_list)
train_lr_list = np.array(train_lr_list)
validation_loss_list = np.array(validation_loss_list)

t0 = time.time()
t0_epoch = time.time()
# max_seconds = 60*60*11 + 60*30 # 11 horas y 30 minutos
max_seconds = 60*60*24*1000 # 1000 días

for epoch in range(EPOCH0, EPOCHS, 1):
    days, hours, minutes, seconds = elapsed_time(t0)
    days_epoch, hours_epoch, minutes_epoch, seconds_epoch = elapsed_time(t0_epoch)
    t0_epoch = time.time()
    print(f"\nEpoch {epoch}\n-------------------------------, Total: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds\tEpoch: {days_epoch} days, {hours_epoch} hours, {minutes_epoch} minutes, {seconds_epoch} seconds")
    train_loss, train_lr, transformer = train_loop(train_dataloader, transformer, loss_function, optimizer, device, mask, epoch)
    validation_loss, accuracy = validation_loop(validation_dataloader, transformer, loss_function, device, mask, epoch)

    train_loss_list = np.append(train_loss_list, train_loss)
    train_lr_list = np.append(train_lr_list, train_lr)
    validation_loss_list = np.append(validation_loss_list, validation_loss)

    if validation_loss < best_loss:
        best_loss = validation_loss
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        # Delete all files in folder
        files = os.listdir(MODEL_PATH)
        for f in files:
            os.remove(os.path.join(MODEL_PATH, f))
        torch.save(transformer.state_dict(), f'{MODEL_PATH}/transformer_{epoch+1}_{actual_step.get_step()}.pth')
        torch.save(transformer, f'{MODEL_PATH}/transformer_model_{epoch+1}_{actual_step.get_step()}.pth')
        val = np.array([validation_loss])
        np.save(f'{MODEL_PATH}/validation_loss_{epoch+1}_{actual_step.get_step()}.npy', val)
        print(f"Best model saved with loss {best_loss} at epoch {epoch}, in {time.time()-t0} ms, with lr {train_lr[-1]} and in step {actual_step.get_step()} --> {MODEL_PATH}/transformer_{epoch+1}_{actual_step.get_step()}.pth")

    days, hours, minutes, seconds = elapsed_time(t0)
    train_elapsed_seconds = elapsed_seconds(days, hours, minutes, seconds)
    if train_elapsed_seconds > max_seconds:
        print("Time out!")
        break

    if LEN_SUBSET_TRAIN == 1 and accuracy == 1.0:
        print("Accuracy 100%!")
        break

np.save(f'{MODEL_PATH}/train_loss_list.npy', train_loss_list)
np.save(f'{MODEL_PATH}/train_lr_list.npy', train_lr_list)
np.save(f'{MODEL_PATH}/validation_loss_list.npy', validation_loss_list)

print("Done!")
