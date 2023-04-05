# def bci2a_eeg_net_t():
#     set_random_seeds(seed=14388341, cuda=cuda)
#     ds = dataset_loader.DatasetFromBraindecode('bci2a', subject_ids=[1])
#     ds.preprocess_dataset(resample_freq=128)
#     windows_dataset = ds.create_windows_dataset(trial_start_offset_seconds=-0.5)
#     n_channels = ds.get_channel_num()
#     input_window_samples = ds.get_input_window_sample()
#     model = nn_models.EEGNetv4(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
#                                kernel_length=64, drop_prob=0.5)
#     if cuda:
#         model.cuda()
#     summary(model, (1, n_channels, input_window_samples, 1))
#     n_epochs = 100
#     lr = 0.001
#     batch_size = 64
#     subjects_windows_dataset = windows_dataset.split('subject')
#     subjects_accuracy = []
#     for subject, windows_dataset in subjects_windows_dataset.items():
#         model = nn_models.EEGNetv4(in_chans=n_channels, n_classes=4, input_window_samples=input_window_samples,
#                                    kernel_length=64, drop_prob=0.5)
#         if cuda:
#             model.cuda()
#         split_by_session = windows_dataset.split('session')
#         train_set = split_by_session['session_T']
#         test_set = split_by_session['session_E']
#         train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         test_dataloader = DataLoader(test_set, batch_size=batch_size)
#         loss_function = nn.NLLLoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - 1)
#         for epoch in range(n_epochs):
#             train_loss = 0.
#             train_accuracy = 0.
#             model.train()
#             for batch in train_dataloader:
#                 x_train, y_train, _ = batch
#                 if cuda:
#                     x_train = x_train.cuda()
#                     y_train = y_train.cuda()
#                 out = model(x_train)
#                 loss = loss_function(out, y_train)
#                 train_loss += loss
#                 predict = torch.max(out, 1)[1]
#                 accuracy = (predict == y_train).sum()
#                 train_accuracy += accuracy
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             scheduler.step()
#             train_loss = train_loss / len(train_dataloader)
#             train_accuracy = train_accuracy / len(train_dataloader)
#
#             valid_loss = 0.
#             valid_accuracy = 0.
#             model.eval()
#             for batch in test_dataloader:
#                 x_test, y_test, _ = batch
#                 if cuda:
#                     x_test = x_test.cuda()
#                     y_test = y_test.cuda()
#                 out = model(x_test)
#                 loss = loss_function(out, y_test)
#                 valid_loss += loss
#                 predict = torch.max(out, 1)[1]
#                 accuracy = (predict == y_test).sum()
#                 valid_accuracy += accuracy
#             valid_loss = valid_loss / len(test_dataloader)
#             valid_accuracy = valid_accuracy / len(test_dataloader)
#             print('Epoch: {} Train Loss: {:.6f} Acc: {:.6f} Valid Loss: {:.6f} Acc: {:.6f}'.format(
#                 epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))
