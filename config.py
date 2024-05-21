import secrets
import os

class Config:
    SECRET_KEY = secrets.token_hex(16)  # Génère une clé secrète aléatoire
    RECAPTCHA_PUBLIC_KEY = '6Lc2TeIpAAAAAM76qT-L0i4K7c80LmaInyxV2yJD'
    RECAPTCHA_PRIVATE_KEY = '6Lc2TeIpAAAAALzLFxYvdmF-zwTabLLBnGNFJbAa'
    RECAPTCHA_USE_SSL = False  # Passez à True si vous utilisez HTTPS
    RECAPTCHA_OPTIONS = {'theme': 'white'}
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
