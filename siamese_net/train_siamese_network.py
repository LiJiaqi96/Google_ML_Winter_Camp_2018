from siamese_network import SiameseNetwork
import numpy as np
import os


def main():
    learning_rate = 10e-4
    batch_size = 32
    epochs = 15
    height = 256
    width = 256
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Dense1'] = 1
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net_lr10e-4'
    siamese_network = SiameseNetwork(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs = epochs,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )

    # Data
    os.chdir("/data8t/ljq/whale_data/whale_data/siamese_networks/")
    train_data_temp = np.load('training_data.npy')
    train_data = [train_data_temp[0]]
    train_data.append(train_data_temp[1])
    train_label = np.load('training_label.npy')
    val_data_temp = np.load('validation_data.npy')
    val_data = [val_data_temp[0]]
    val_data.append(val_data_temp[1])
    val_label = np.load('validation_label.npy')
    # train_base_num = 1000
    # train_data = [np.zeros((train_base_num * 6, height, width, 3)) for j in range(2)]
    # train_label = np.zeros((train_base_num * 6, 1))
    # val_data = [np.ones((train_base_num * 2, height, width, 3)) for j in range(2)]
    # val_label = np.ones((train_base_num * 2, 1))

    siamese_network.train_siamese_network(model_name='siamese_net_whale',train_data=train_data,
                                                                train_label=train_label,val_data = val_data,
                                                                val_label = val_label)

if __name__ == "__main__":
    main()
