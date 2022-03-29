
{
   // Local Variables
  local transformer_model = "facebook/bart-base", // pre-trained bert model to be used, obtained from huggingface.com
  local transformer_max_length = 1024,
  local transformer_hidden_size = 768,
  // Dataset Reader: Here we can define constructor parameters of dataset reader that we are using. "type" key is
  // important as it decides which dataset reader we are using. You can find "stance_data_reader" defined just above
  // class definition of StanceDataReader(modules/readers/StanceDataReader.py Line 25). Other keys are constructor
  // parameters.
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
  "train_data_path": '_@train', // We download data directly from web
  "validation_data_path": '_@val',
  // Model: Here we define which model to use and the constructor parameters of that model. We use "crf_tagger" which is
  // provided by allennlp (http://docs.allennlp.org/v0.9.0/api/allennlp.models.crf_tagger.html). This model requires an
  // encoder on top and we use Bi-LSTM.
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
    "num_epochs": 5,
    "grad_norm": 7.85,
    "patience": 4,
    // Callback: we use Weights and Bias to log out training. We use custom created W&B callback.
"callbacks": [
        {
        "type": "custom_wandb",
         'entity': std.extVar('WANDB_ENTITY'),
         'project': std.extVar('WANDB_PROJECT'),
         'files_to_save': ["config.json", "out.log","metrics.json"],
         },
         ],
  }
}