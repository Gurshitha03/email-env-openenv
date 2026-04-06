from openenv.server import create_app
from my_env import EmailEnv

app = create_app(EmailEnv)
