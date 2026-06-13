# ⚖️ CLT RAG Chatbot

Chatbot especialista na **Consolidação das Leis do Trabalho (CLT)** brasileira, desenvolvido com RAG agêntico usando Gemini 2.5 Flash, ChromaDB e LangGraph.

Projeto acadêmico — Disciplina de NLP, 6º Semestre, Data Science.

🔗 **[Acesse o chatbot aqui](https://clt-rag-chatbot-q6672mji7webv8wr6qhznf.streamlit.app/)**

---

## Sobre o Projeto

O sistema responde perguntas sobre direitos trabalhistas em português, buscando trechos relevantes diretamente na CLT e gerando respostas precisas com citação dos artigos. O pipeline RAG agêntico avalia a qualidade dos chunks recuperados e reformula a query automaticamente quando a relevância está abaixo do limiar.

---

## Funcionalidades da Interface

A interface Streamlit oferece três controles principais para o usuário:

| Recurso | Descrição |
|---|---|
| **Toggle de modo agêntico** | Permite alternar entre o pipeline agêntico completo (com avaliação de relevância e retry de query) e o pipeline simples (uma única chamada à API, mais rápido). |
| **Toggle de exibição de artigos** | Quando ativo, exibe um painel expansível abaixo de cada resposta listando os artigos da CLT efetivamente utilizados, com trecho legível para conferência. |
| **Tooltips em menções de artigos** | Sempre que o LLM menciona um artigo no corpo da resposta (ex: "Art. 130"), basta passar o mouse sobre o texto para visualizar o trecho do artigo correspondente. |

---

## Arquitetura

```
PDF da CLT → Extração → Chunking por artigo → Embeddings → ChromaDB
                                                                ↓
                              Pergunta → Condense → Retrieval (k=8)
                                                        ↓
                                               Avaliação de relevância
                                            (cosine similarity + LLM judge)
                                                        ↓
                                         relevância ok? → Gemini → Resposta
                                         relevância baixa? → Reformula query → Retrieval
```

---

## Stack Tecnológica

| Componente    | Tecnologia                              | Motivo                                      |
|---------------|-----------------------------------------|---------------------------------------------|
| LLM           | Gemini 2.5 Flash (Google AI Studio)     | Gratuito, alta qualidade em português       |
| Embeddings    | `models/text-embedding-004` (Google)    | Mesma API, sem custo extra                  |
| Vectorstore   | ChromaDB (persistido no repositório)    | Sem infraestrutura, todos acessam via GitHub|
| Framework     | LangChain + LangGraph                   | Padrão acadêmico, bem documentado           |
| Interface     | Streamlit Community Cloud               | Deploy gratuito, URL pública                |
| Versionamento | GitHub                                  | Tracking de tarefas via Issues              |
| Documento     | CLT em PDF                              | Domínio público, bem estruturado            |

---

## Estrutura do Repositório

```
clt-rag-chatbot/
├── data/
│   ├── raw/                  # PDF original da CLT
│   ├── processed/            # Texto extraído
│   └── vectorstore/          # ChromaDB persistido
├── notebooks/
│   ├── 01_chunking_experiments.ipynb   # Estratégia de chunking (Pessoa 1)
│   └── 02_rag_evaluation.ipynb         # Avaliação RAG agêntico (Pessoa 2)
├── src/
│   ├── ingestion/            # Extração, chunking e embeddings (Pessoa 1)
│   │   ├── download_clt.py
│   │   ├── extractor.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/            # Pipeline RAG agêntico (Pessoa 2)
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   ├── relevance.py
│   │   ├── chain.py          # RAG simples (1 chamada) + get_answer_with_sources
│   │   └── agent.py          # RAG agêntico (LangGraph) + get_agent_answer_with_sources
│   └── app/                  # Interface Streamlit (Pessoa 3)
│       └── streamlit_app.py  # Toggles, expander de artigos e tooltips
├── tests/
│   └── questions_benchmark.json        # Conjunto de perguntas para avaliação
├── .streamlit/
│   └── config.toml           # Configuração de tema da interface
├── .env.example
├── requirements.txt
└── README.md
```

---

## Como Rodar Localmente

### Pré-requisitos

- Python 3.11
- Microsoft Visual C++ Redistributable ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe))
- Chave de API do Google AI Studio ([obter gratuitamente](https://aistudio.google.com/apikey))

### 1. Clonar o repositório

```bash
git clone https://github.com/eduardogca/clt-rag-chatbot.git
cd clt-rag-chatbot
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
cp .env.example .env
```

Edite o `.env` e insira sua chave:

```
GOOGLE_API_KEY=sua_chave_aqui
```

### 4. Rodar a interface

```bash
# Windows
set PYTHONPATH=. && py -3.11 -m streamlit run src/app/streamlit_app.py

# Linux / Mac
PYTHONPATH=. streamlit run src/app/streamlit_app.py
```

> **Nota:** o ChromaDB já está persistido no repositório em `data/vectorstore/`. Não é necessário rodar o pipeline de ingestão novamente.

---

## Pipeline de Ingestão (referência)

Caso precise regenerar o vectorstore do zero:

```bash
python src/ingestion/extractor.py    # extrai texto do PDF
python src/ingestion/chunker.py      # divide por artigo com metadados
python src/ingestion/embedder.py     # gera embeddings e salva no ChromaDB
```

O chunking foi feito por artigo (1.183 chunks), com metadados de artigo e seção para citação precisa nas respostas.

---

## Pipeline RAG: Agêntico × Simples

O sistema implementa dois pipelines de geração de resposta, selecionáveis em tempo de execução pelo toggle na interface.

### Modo Agêntico (`src/retrieval/agent.py`)

Grafo LangGraph com avaliação de relevância em dois estágios:

1. **Retrieval** — recupera k=8 chunks do ChromaDB
2. **Avaliação** — pontuação combinada: `0.4 × cosine_similarity + 0.6 × llm_judge`
3. **Decisão** — se score médio < 0.45 e tentativas < 3, reformula a query e repete
4. **Geração** — usa apenas chunks com score ≥ 0.45 para gerar a resposta

Mais preciso, porém faz múltiplas chamadas à API (~9 por pergunta no pior caso).

### Modo Simples (`src/retrieval/chain.py`)

Pipeline direto sem avaliação de relevância:

1. **Retrieval** — recupera k=6 chunks do ChromaDB
2. **Geração** — passa todos os chunks ao LLM e gera a resposta

Mais rápido (~10s por pergunta) e econômico em quota da API. Ideal para uso casual.

---

## Avaliação

Os testes documentados em `notebooks/02_rag_evaluation.ipynb` cobrem três métricas complementares sobre um benchmark de 10 perguntas:

| Métrica | Resultado | O que mede |
|---|---|---|
| **Keyword Accuracy** | 10/10 (100%) | Resposta contém as palavras-chave esperadas |
| **Recall@6** | 8/8 (100%) | Artigo esperado está entre os 6 chunks recuperados |
| **Rastreabilidade** | 10/10 | Artigo correto foi utilizado na geração da resposta |

---

## Divisão de Tarefas

| Pessoa | Responsabilidade | Módulo |
|---|---|---|
| Felipe Teodoro | Data pipeline: extração, chunking e embeddings | `src/ingestion/` + `notebooks/01` |
| Pessoa 2 | RAG core: retrieval, prompts, avaliação e LangGraph | `src/retrieval/` + `notebooks/02` |
| Pessoa 3 | Interface Streamlit, deploy e documentação | `src/app/` + `README` |

---

## Exemplos de Perguntas

- "Quantos dias de férias o trabalhador tem direito por ano?"
- "O empregador pode demitir durante licença médica?"
- "Qual a jornada máxima de trabalho diária?"
- "Quais são os direitos da trabalhadora gestante?"
- "O que é aviso prévio e qual o prazo mínimo?"
- "Como funciona o FGTS?"

---

## Licença

Projeto acadêmico sem fins comerciais. A CLT é documento de domínio público.
