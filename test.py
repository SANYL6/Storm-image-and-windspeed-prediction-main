from storm_tasks.task2_CNN.train import Train_Validate


surprise_model = Train_Validate('/home/jxy/Storm-Prediction-main/tst', task='WindSpeed', device='cuda',
                                        batch_size_train=135, batch_size_val=100, batch_size_test=100,
                                        lr=2e-11, epoch=30, weight_decay=1e-1, split_method='random', num_storms=1, surprise_storm=True,
                                    resume=True, resume_path='/home/jxy/Storm-Prediction-main/storm_tasks/task2_CNN/CNNGeneral_epoch_6__2024-06-13_08-36-41.pth')
surprise_model.draw_result(type='predict')

# surprise_model.predict()

# storm_list=['hig', 'gme', 'woc', 'blq', 'kqu', 'wsy', 'ipa', 'ztb', 'qpq', 'pjj']
#
# cnn_model = Train_Validate('/home/jxy/Storm-Prediction-main/tst', task='WindSpeed', device='cuda',
#                                         batch_size_train=32, batch_size_val=1000, batch_size_test=1000,
#                                         lr=2e-3, epoch=20, split_method='random',resume=False,resume_path='CNNGeneral_epoch_16.pth')
#
# cnn_model.train_whole()
#
# torch.save({'model_state_dict': cnn_model.model.state_dict(),
#             'optimizer_state_dict': cnn_model.optimizer.state_dict(),
#             }, 'general_model.pth')

# surprise_model = Train_Validate('/root/autodl-tmp/Storm-preiction/tst', task='WindSpeed', device='cuda',
#                                         batch_size_train=32, batch_size_val=100, batch_size_test=100,
#                                         lr=2e-8, epoch=50, split_method='random', num_storms=3, surprise_storm=True,
#                                         resume=True, resume_path='general_model.pth')
#
# surprise_model.draw_result()
# surprise_model.draw_result()
