
{
  local transformer_model = "bert-base-uncased",
  local transformer_max_length = 512,
  //local transformer_max_length = 128,
  local transformer_hidden_size = 768,
  "dataset_reader": {
    "type": "stance_data_reader",
    "task":1,
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
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": false,
    "dropout": 0.25,
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
        "input_size": transformer_hidden_size,
        "hidden_size": 400,
        "num_layers": 4,
        "dropout": 0.6,
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
        "lr": 0.0001
    },
    "validation_metric": "+accuracy",
    "num_epochs": 30,
    "grad_norm": 7.85,
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