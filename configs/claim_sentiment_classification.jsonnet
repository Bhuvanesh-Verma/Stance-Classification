
{
  local transformer_model = "distilbert-base-uncased-finetuned-sst-2-english",
  local transformer_max_length = 512,
  local transformer_hidden_size = 768,
  "dataset_reader": {
    "type": "sent_data_reader",
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
  "train_data_path": 'Data/sentiment_train.txt',
  "validation_data_path": 'Data/sentiment_val.txt',
  "model": {
    "type": "basic_classifier",
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
    "seq2vec_encoder": {
        "type": "lstm",
        "input_size": transformer_hidden_size,
        "hidden_size": 300,
        "num_layers": 2,
        "dropout": 0.4394,
        "bidirectional": true
    },
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.00001,
    },
    "validation_metric": "+accuracy",
    "num_epochs": 10,
    "grad_norm": 7.0,
    "patience": 5,
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