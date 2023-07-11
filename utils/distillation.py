import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


# Defined functions
def remove_layers(model, layers):

  oldModuleList = model.base_model.encoder.layer
  newModuleList = nn.ModuleList()
  
  # Remove the selected layers
  for i in range(0, len(oldModuleList)):
    if i not in layers:
      newModuleList.append(oldModuleList[i])

  model.base_model.encoder.layer = newModuleList
  model.config.num_hidden_layers = len(oldModuleList) - len(layers)


def distil_model(path, s_model, t_model, args, train_data, val_data, checkpoint):
    
  trainer = DistilTrainer(
    model=s_model,
    teacher=t_model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
  )
  
  trainer.train(resume_from_checkpoint=checkpoint)
  
  if path is not None:
  
    if not os.path.isdir(path):
      os.makedirs(path)
  
    # Save the model
    s_model.config.output_hidden_states = False
    s_model.save_pretrained(path)

    # Save the model's loss
    train_epoch, train_loss = [], []
    eval_epoch, eval_loss = [], []
  
    with open(os.path.join(path, 'scores.txt'),'w') as file:
    
      for obj in trainer.state.log_history:
      
        if 'loss' in obj:
          train_epoch.append(obj['epoch'])
          train_loss.append(obj['loss'])
      
        if 'eval_loss' in obj:
          eval_epoch.append(obj['epoch'])
          eval_loss.append(obj['eval_loss'])
      
        file.write(json.dumps(obj))
        file.write('\n')

    # Plot the model's loss
    plt.figure(figsize=(10, 6))
  
    plt.plot(train_epoch, train_loss)
    plt.plot(eval_epoch, eval_loss)
    plt.title('Loss against Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
  
    plt.savefig(os.path.join(path, 'plots.png'))


# Defined classes
class DistillationArguments(TrainingArguments):

  def __init__(self, *args, temperature=2.0, alpha_kld=0.5, alpha_mlm=0.5, alpha_cos=0.5, **kwargs):
    super().__init__(*args, **kwargs)
    self.temperature = temperature
    self.alpha_kld = alpha_kld
    self.alpha_mlm = alpha_mlm
    self.alpha_cos = alpha_cos


class DistilTrainer(Trainer):

  def __init__(self, *args, teacher, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.teacher = teacher
    self._move_model_to_device(self.teacher, self.model.device)
    self.teacher.eval()

  def compute_loss(self, model, inputs, return_outputs=False):
  
    s_outputs = model(**inputs)
    s_logits, s_hidden_states = s_outputs.logits, s_outputs.hidden_states[-1]

    with torch.no_grad():
      t_outputs = self.teacher(**inputs)
      t_logits, t_hidden_states = t_outputs.logits, t_outputs.hidden_states[-1]
    
    # Calculate the Kullback-Leibler divergence loss
    kld_loss_fct = nn.KLDivLoss(reduction='batchmean')
  
    loss_kld = (
      kld_loss_fct(
          F.log_softmax(s_logits / self.args.temperature, dim=-1),
          F.softmax(t_logits / self.args.temperature, dim=-1)
      ) 
      * (self.args.temperature) ** 2
    )
    
    # Calculate the masked-language modelling loss
    loss_mlm = s_outputs.loss
    
    # Calculate the cosine embedding loss
    cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
    
    dim = s_hidden_states.size(-1)
    s_hidden_states = s_hidden_states.view(-1, dim)
    t_hidden_states = t_hidden_states.view(-1, dim)
    
    target = s_hidden_states.new(s_hidden_states.size(0)).fill_(1)
    loss_cos = cosine_loss_fct(s_hidden_states, t_hidden_states, target)
  
    loss = self.args.alpha_kld * loss_kld + self.args.alpha_mlm * loss_mlm + self.args.alpha_cos * loss_cos
    return (loss, s_outputs) if return_outputs else loss
