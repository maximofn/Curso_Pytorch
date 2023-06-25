import torch
from transformers import BertModel, BertTokenizer

# class CustomBertModel(BertModel):
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         if position_ids is None:
#             position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
#             position_ids = position_ids.unsqueeze(0).expand(input_shape)
#         else:
#             position_ids = position_ids.expand(input_shape)

#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            
#         embeddings = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds
#         )

#         token_embeddings = embeddings.detach().clone() # Guarda los embeddings de los tokens
#         positional_embeddings = self.embeddings.position_embeddings(position_ids).detach().clone() # Guarda los embeddings posicionales
#         embeddings_sum = token_embeddings + positional_embeddings

#         return super().forward(
#             inputs_embeds=embeddings,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=None,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         ), token_embeddings, positional_embeddings, embeddings_sum

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = CustomBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# sentence = "I gave the dog a bone because it was hungry"
# # sentence = "I gave the dog a bone because it was old"
# input_ids = tokenizer.encode(sentence, add_special_tokens=True)
# input_ids_tensor = torch.tensor([input_ids])

# tokens = tokenizer.convert_ids_to_tokens(input_ids)

# with torch.no_grad():
#     outputs, token_embeddings, positional_encodings, embeddings_with_positional_encoding = model(input_ids_tensor)

# print(f"Frase de entrada: {sentence}")
# print(f"Tokens: {tokens}")
# print(f"IDs de los tokens: {input_ids}")
# for i, token in enumerate(tokens):
#     print(f"El token \"{token}\" es el número {i} y tiene el ID {input_ids[i]}")
# print(f"\nToken embeddings: {token_embeddings.shape}")
# print(token_embeddings)
# print(f"\nPositional encodings: {positional_encodings.shape}")
# print(positional_encodings)
# print(f"\nEmbeddings with positional encoding: {embeddings_with_positional_encoding.shape}")
# print(embeddings_with_positional_encoding)
# print(f"\nX shape: {embeddings_with_positional_encoding.shape}")
# print(f"\nX·X^T shape: {torch.matmul(embeddings_with_positional_encoding, embeddings_with_positional_encoding.transpose(1, 2)).shape}")
# print(f"\nX·X^T: {torch.matmul(embeddings_with_positional_encoding, embeddings_with_positional_encoding.transpose(1, 2))}")
# print(f"\nlen tokens: {len(tokens)}")


def extract_embeddings(input_sentence, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
    input_ids_tensor = torch.tensor([input_ids])
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids_tensor)
        
    token_embeddings = outputs[0][0]
    
    # Los embeddings posicionales están en la segunda capa de los embeddings de la arquitectura BERT
    positional_encodings = model.embeddings.position_embeddings.weight[:len(input_ids), :].detach()
    
    embeddings_with_positional_encoding = token_embeddings + positional_encodings
    
    return tokens, input_ids, token_embeddings, positional_encodings, embeddings_with_positional_encoding


sentence = "I gave the dog a bone because it was hungry"
# sentence = "I gave the dog a bone because it was old"
tokens, input_ids, token_embeddings, positional_encodings, embeddings_with_positional_encoding = extract_embeddings(sentence)

print(f"Frase de entrada: {sentence}")
print(f"Tokens: {tokens}")
print(f"IDs de los tokens: {input_ids}")
for i, token in enumerate(tokens):
    print(f"El token \"{token}\" es el número {i} y tiene el ID {input_ids[i]}")
print(f"\nToken embeddings: {token_embeddings.shape}")
print(token_embeddings)
print(f"\nPositional encodings: {positional_encodings.shape}")
print(positional_encodings)
print(f"\nEmbeddings with positional encoding: {embeddings_with_positional_encoding.shape}")
print(embeddings_with_positional_encoding)
print(f"\nX shape: {embeddings_with_positional_encoding.shape}")
print(f"\nX·X^T shape: {torch.matmul(embeddings_with_positional_encoding, embeddings_with_positional_encoding.transpose(0, 1)).shape}")
print(f"\nX·X^T: {torch.matmul(embeddings_with_positional_encoding, embeddings_with_positional_encoding.transpose(0, 1))}")
print(f"\nlen tokens: {len(tokens)}")