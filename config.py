config = {
    'ACTION_DIM': 14,
    'STATE_DIM': [28, 28],
    'MAX_ITER': 10000,
    'LR': 0.0001,
    'CLIP': 0.2,
    'GAMMA': 0.95,
    'C1': 1.0,      # value function parameter in loss
    'C2': 0.0,      # entropy bonus parameter in loss

    'MAX_T': 20,
    'MAX_ITERATION': 10000,
    'T': 100,
    'TRAIN_EPOCH': 5,
    'BATCH_SIZE': 64,

    'SAVED_GAN': './saved_gan/2',
    'SAVED_CNN': './saved_cnn'
}