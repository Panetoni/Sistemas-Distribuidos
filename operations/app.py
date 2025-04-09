import random
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import requests
import os
from torchvision import models, transforms
import torch
from scipy.spatial.distance import cosine
import numpy as np

VECTOR_DATABASE_URL = "http://vector_database:5000"

import random
from PIL import ImageDraw

def riscar_imagem(imagem):
    """Adiciona múltiplos riscos na imagem."""
    draw = ImageDraw.Draw(imagem)
    largura, altura = imagem.size

    # Número aleatório de riscos horizontais e verticais
    num_riscos = random.randint(3, 20)  # Define o número de riscos (entre 3 e 6)

    # Adiciona riscos horizontais
    for _ in range(num_riscos):
        y = random.randint(0, altura)  # Define uma posição aleatória para o risco horizontal
        draw.line((0, y, largura, y), fill="red", width=7)  # Linha horizontal

    # Adiciona riscos verticais
    for _ in range(num_riscos):
        x = random.randint(0, largura)  # Define uma posição aleatória para o risco vertical
        draw.line((x, 0, x, altura), fill="blue", width=7)  # Linha vertical

    return imagem


def espelhar_imagem(imagem):
    """Espelha a imagem horizontalmente."""
    return ImageOps.mirror(imagem)

def inverter_imagem(imagem):
    """Inverte a imagem verticalmente."""
    return ImageOps.flip(imagem)

def alterar_opacidade(imagem, opacidade=0.8):
    """Altera a opacidade da imagem."""
    if imagem.mode != "RGBA":
        imagem = imagem.convert("RGBA")
    alpha = imagem.split()[-1]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacidade)
    imagem.putalpha(alpha)
    return imagem

def aplicar_transformacoes(imagem):
    """Aplica todas as transformações à imagem."""
    transformacoes = [riscar_imagem, espelhar_imagem, inverter_imagem, alterar_opacidade]
    
    # Aplica todas as transformações na ordem
    for transformacao in transformacoes:
        if transformacao == alterar_opacidade:
            imagem = transformacao(imagem, opacidade=random.uniform(0.3, 0.8))  # Opacidade aleatória
        else:
            imagem = transformacao(imagem)
    
    return imagem

def gerar_vector(imagem):
    if imagem.mode == 'RGBA':
        imagem = imagem.convert('RGB')

    # Aplica transformações aleatórias
    imagem = aplicar_transformacoes(imagem)
    
    # Garante que a imagem esteja no formato RGB após as transformações
    if imagem.mode == 'RGBA':
        imagem = imagem.convert('RGB')
    
    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagem_transformada = transformacao(imagem).unsqueeze(0)
    
    modelo = models.resnet50(weights="IMAGENET1K_V1")
    modelo.eval()
    with torch.no_grad():
        vetor = modelo(imagem_transformada)
    return vetor.squeeze().numpy().tolist()

def adicionar_vector(name, vector):
    response = requests.post(f"{VECTOR_DATABASE_URL}/vector", json={"name": name, "vector": vector})
    if response.status_code == 201:
        print(f"Vetor adicionado: {name}")
    else:
        print(f"Erro ao adicionar vetor: {response.text}")

def buscar_vetores():
    response = requests.get(f"{VECTOR_DATABASE_URL}/vectors")
    if response.status_code == 200:
        database = response.json()
        return [(name, np.array(vector)) for name, vector in database]
    else:
        print("Erro ao buscar vetores.")
        return []

def processar_e_comparar(diretorio_imagens):
    imagens = [img for img in os.listdir(diretorio_imagens) if img.endswith(".jpg")]
    database = buscar_vetores()
    total_acertos = 0

    for imagem_nome in imagens:
        caminho_imagem = os.path.join(diretorio_imagens, imagem_nome)
        imagem = Image.open(caminho_imagem)
        
        vetor_imagem = gerar_vector(imagem)
        adicionar_vector(imagem_nome, vetor_imagem)
        
        menor_distancia = float('inf')
        imagem_mais_parecida = None
        for nome, vetor in database:
            distancia = cosine(vetor_imagem, vetor)
            #print(f"Distância entre os vetores: {distancia:.4f}")
            if distancia < menor_distancia:
                menor_distancia = distancia
                imagem_mais_parecida = nome

        # Verifica se a imagem mais parecida é a própria imagem
        if imagem_nome == imagem_mais_parecida:
            total_acertos += 1

        print(f"Imagem: {imagem_nome}, Mais Parecida: {imagem_mais_parecida}, Distância: {menor_distancia:.4f}")

    # Calcula a precisão
    total_imagens = len(imagens)
    precisao = total_acertos / total_imagens if total_imagens > 0 else 0
    print(f"\nPrecisão: {precisao:.2%} ({total_acertos}/{total_imagens})")

if __name__ == "__main__":
    diretorio_imagens = "./imagens"
    processar_e_comparar(diretorio_imagens)
