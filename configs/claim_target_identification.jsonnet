
{
  local transformer_model = "facebook/bart-base",
  local transformer_max_length = 1024,
  //local transformer_max_length = 128,
  local transformer_hidden_size = 768,
  "dataset_reader": {
    "type": "sequence_tagging",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": transformer_max_length
      },
      },
    },
  "train_data_path": 'Data/train.txt',
  "test_data_path": 'Data/val.txt',
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": false,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": transformer_max_length,
            //"train_parameters": true
            "train_parameters": false
        },
       },
    },
    "encoder": {
        "type": "lstm",
        //"input_size": 50 + 128,
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
        "lr": 0.005
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
         'files_to_save': ["config.json", "out.log","metrics.json"]
         },
         ],
  }
}