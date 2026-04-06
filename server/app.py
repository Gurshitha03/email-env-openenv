from openenv.server import create_app
from my_env import EmailEnv

app = create_app(EmailEnv)

def main():
    return app

if __name__ == "__main__":
    main()
