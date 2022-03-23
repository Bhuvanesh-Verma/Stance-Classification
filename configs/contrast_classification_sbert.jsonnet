
{
  local transformer_model = "sentence-transformers/all-mpnet-base-v2",
  local transformer_max_length = 514,
  local transformer_hidden_size = 768,
  "dataset_reader": {
    "type": "stance_data_reader",
    "task": 3,
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
  "train_data_path": '_@train',
  "validation_data_path": '_@val',
  "model": {
   "type": "relation_classifier",
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
        "hidden_size": 64,
        "num_layers": 3,
        "dropout": 0.3,
        "bidirectional": true
    }


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
    "num_epochs": 10,
    "grad_norm": 7.0,
    "patience": 4,
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