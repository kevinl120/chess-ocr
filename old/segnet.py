from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="data/segnet_test")
trainer.setTrainConfig(object_names_array=["board"], batch_size=4, num_experiments=100)
trainer.trainModel()