#!/usr/bin/env python3
"""
Script de inicialização do Stock Forecast Service
Executa setup inicial: instala dependências, treina modelos, etc.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str):
    """Executa um comando e mostra o resultado"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro em {description}: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False

def main():
    """Função principal de setup"""
    print("🚀 Iniciando setup do Stock Forecast Service...")

    # Verifica se está no diretório correto
    if not Path("requirements.txt").exists():
        print("❌ Execute este script do diretório raiz do projeto!")
        sys.exit(1)

    # Instala dependências
    if not run_command("pip install -r requirements.txt", "Instalando dependências"):
        sys.exit(1)

    # Treina modelos (opcional)
    print("\n🤖 Deseja treinar os modelos agora? (recomendado)")
    print("Isso pode levar alguns minutos...")
    train = input("Treinar modelos? (y/N): ").lower().strip()

    if train == 'y':
        if not run_command("python -m ml.train", "Treinando modelos ML"):
            print("⚠️ Treinamento falhou, mas você pode executar manualmente depois")
    else:
        print("ℹ️ Pule o treinamento. Execute 'python -m ml.train' quando quiser treinar os modelos.")

    # Executa testes
    print("\n🧪 Executando testes...")
    if not run_command("python -m pytest tests/ -v", "Executando testes"):
        print("⚠️ Alguns testes falharam. Verifique os logs acima.")

    print("\n🎉 Setup concluído!")
    print("\n📋 Próximos passos:")
    print("1. Execute a API: python -m app.main")
    print("2. Execute o dashboard: streamlit run dashboard.py")
    print("3. Acesse: http://localhost:8000/docs (API) e http://localhost:8501 (Dashboard)")

if __name__ == "__main__":
    main()