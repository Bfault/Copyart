#!/usr/bin/env python3

import secrets

def build_env():
    env = {
        'KAGGLE': {
            'KAGGLE_AUTHOR': "ikarus777",
            'KAGGLE_NAME': "best-artworks-of-all-time",
            'KAGGLE_USERNAME': "INSERT_YOUR_KAGGLE_USERNAME_HERE",
            'KAGGLE_KEY': "INSERT_YOUR_KAGGLE_API_KEY_HERE"
        },
    }

    with open('.env', 'w') as f:
        for key, value in env.items():
            f.write(f"#{key}\n")
            for k, v in value.items():
                f.write(f"{k}={v}\n")

if __name__ == "__main__":
    build_env()