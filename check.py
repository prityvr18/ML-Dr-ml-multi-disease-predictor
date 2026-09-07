from src.training.config.settings import Settings


settings = Settings()


log_path = settings.log_path
diabes_data = settings.diabetes_dataset_path


print(log_path)
print(diabes_data)


