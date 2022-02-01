
{
  local transformer_model = "BERT-for-RRC-ABSA/pt_model/laptop_pt",
  local transformer_max_length = 512,
  local transformer_hidden_size = 768,
  "dataset_reader": {
    "type": "stance_data_reader",
    "task": 2,
    "tokenizer": {
    "type": "pretrained_transformer",
    "model_name": transformer_model,
    "add_special_tokens": false
  },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": transformer_max_length
      },
      },
    },
  "train_data_path": 'Data/sentiment_train.txt@train',
  "validation_data_path": 'Data/sentiment_val.txt@val',
  "model": {
   "type": "bert_for_classification",
    "dropout": 0.5,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": transformer_max_length,
            "train_parameters": false
        },
       },
    },

  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
        "type": "adamw",
        "lr": 3e-5,
    },
    "validation_metric": "+accuracy",
    "num_epochs": 5,
    "grad_norm": 7.0,
    "patience": 3,
"callbacks": [
        {
        "type": "custom_wandb",
         'entity': std.extVar('WANDB_ENTITY'),
         'project': std.extVar('WANDB_PROJECT'),
         'files_to_save': ["config.json", "out.log","metrics.json"],
         //'serialization_dir': 'experiment/wandb'
         },
         ],
  }
}