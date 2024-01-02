sweep_configuration = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'valid_accuracy'},
    'parameters': {
        'lr': {'min': 0.0001, 'max': 0.01},
        'dropout_ratio': {'values': [0.1, 0.2, 0.3]},
        'weight_decay': {'min': 0.00001, 'max': 0.01}
        }
}

def run_sweep():
  num_epochs = 100
  patience = 3
  model_name = 'exp1'

  run = wandb.init(project = 'test-mnist')

  model = CNN(num_classes = 10, dropout_ratio = wandb.config.dropout_ratio)
  model.weight_initialization()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr = wandb.config.lr, weight_decay = wandb.config.weight_decay)

  # wandb에 모델의 weight & bias, graident를 시각화합니다.
  run.watch(model, criterion, log = 'all')
  return training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name, run)


sweep_id = wandb.sweep(
      sweep = sweep_configuration,
      project = 'test-mnist'
)

wandb.agent(sweep_id, function = run_sweep, count = 10)