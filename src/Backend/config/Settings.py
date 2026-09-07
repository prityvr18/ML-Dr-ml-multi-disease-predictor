# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     log_path: str
#     diabetes_model_path: str
#     heart_disease_model_path: str
#     diabetes_dataset_path: str = ""
#     heart_disease_dataset_path: str = ""
#     hyper_params_yaml_path: str = ""
#     diabetes_target_col: str = ""
#     heart_disease_target_col: str = ""
#     test_size: float = 0.2
#     random_state: int = 42

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         extra = "allow"



from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    log_path: str
    diabetes_model_path: str
    heart_disease_model_path: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        
