# --- Importando nossas Ferramentas ---
import os
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- Configuração Inicial ---
load_dotenv()

# --- Nossa "Bancada de Trabalho" Global ---
# Elas começarão vazias e serão preenchidas quando o app iniciar.
embedding_model = None
qdrant_client = None
client = None
wines_data_loaded = False


# --- Função de "Inauguração" (Tudo que acontece quando o App Liga) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, qdrant_client, client, wines_data_loaded
    print("--- Iniciando o Aplicativo RAG Local ---")

    # 1. Carrega o modelo de embedding
    print("Carregando modelo de embedding (pode levar um momento)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("✅ Modelo de embedding carregado.")

    # 2. Cria o cliente Qdrant e a "gaveta" de vinhos
    qdrant_client = QdrantClient(":memory:")
    qdrant_client.recreate_collection(
        collection_name="vinhos_collection",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    print("✅ Cliente Qdrant e coleção 'vinhos_collection' prontos.")

    # 3. Carrega os dados dos vinhos e os insere no Qdrant
    print("Carregando dados dos vinhos do arquivo 'wine-ratings.csv'...")
    try:
        # Pega o caminho do diretório onde o app.py está (a pasta webapp)
        script_dir = os.path.dirname(__file__)
        # Pega o caminho do diretório pai (a raiz do projeto)
        project_root = os.path.dirname(script_dir)
        # Junta o caminho da raiz com o nome do arquivo CSV para formar o caminho completo
        csv_path = os.path.join(project_root, 'wine-ratings.csv')

        print(f"Tentando abrir o CSV em: {csv_path}") # Linha de depuração
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Pula o cabeçalho
            # Formata os dados para o RAG, pegando colunas importantes
            wines_data = [f"Vinho: {row[1]}, Região: {row[2]}, Variedade: {row[3]}, Notas: {row[5]}" for row in reader]

        print(f"Gerando embeddings para {len(wines_data)} vinhos...")
        qdrant_client.upload_records(
            collection_name="vinhos_collection",
            records=[
                models.Record(id=idx, vector=embedding_model.encode(line).tolist(), payload={"texto": line})
                for idx, line in enumerate(wines_data)
            ]
        )
        wines_data_loaded = True
        print("✅ Dados dos vinhos carregados com sucesso no Qdrant!")
    except FileNotFoundError:
        print("❌ ERRO: O arquivo 'wine-ratings.csv' não foi encontrado! Verifique se ele está na pasta principal do projeto.")
    except Exception as e:
        print(f"❌ ERRO ao carregar os dados dos vinhos: {e}")

    # 4. Configura o cliente para conversar com o Ollama
    client = OpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print("✅ Cliente configurado para conversar com o Ollama.")
    print("--- Aplicativo pronto para receber requisições! ---")
    
    yield # O aplicativo fica rodando aqui

    print("--- Encerrando o aplicativo ---")


# --- Criação do Aplicativo FastAPI com o Lifespan ---
app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    prompt: str

def search_in_qdrant(query_text: str):
    if not wines_data_loaded:
        return "Contexto indisponível (arquivo de vinhos não carregado)."
    
    query_vector = embedding_model.encode(query_text).tolist()
    search_result = qdrant_client.search(
        collection_name="vinhos_collection",
        query_vector=query_vector,
        limit=3
    )
    context = "\n".join([hit.payload['texto'] for hit in search_result])
    return context

@app.post("/ask")
def ask_question(request: AskRequest):
    print(f"\nRecebida pergunta: {request.prompt}")
    context = search_in_qdrant(request.prompt)
    print(f"Contexto encontrado:\n{context}")
    
    prompt_final = f"""Use apenas o contexto fornecido para responder à pergunta.
    Contexto: {context}
    
    Pergunta: {request.prompt}
    Resposta:
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME"),
            messages=[
                {"role": "system", "content": "Você é um assistente especialista em vinhos que responde em português."},
                {"role": "user", "content": prompt_final},
            ],
            temperature=0.7,
        )
        resposta_final = response.choices[0].message.content
        print(f"Resposta do LLM: {resposta_final}")
        return {"response": resposta_final}
    except Exception as e:
        print(f"❌ Erro ao chamar o LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))