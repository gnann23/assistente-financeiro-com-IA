"""
Assistente Financeiro IA — Experiência digital de relacionamento financeiro
=============================================================================

Desafio final: uma experiência conversacional guiada por IA generativa,
aplicando boas práticas de UX, para apoiar clientes de uma instituição
financeira. O assistente entende a intenção da mensagem do usuário
(compreensão de linguagem natural), responde de forma contextualizada,
lembra do que já foi dito na conversa (persistência de contexto) e realiza
simulações financeiras simples e seguras.

Funcionalidades:
    - FAQ inteligente sobre produtos financeiros (conta, cartão, empréstimo,
      investimentos, Pix)
    - Simulações financeiras demonstrativas (juros compostos e financiamento)
    - Reconhecimento de intenção por similaridade de texto (TF-IDF + cosseno)
    - Persistência de contexto: memoriza o nome do usuário e o que já foi
      perguntado, e conduz simulações em várias etapas (slot filling)
    - Respostas com avisos de segurança (não é consultoria financeira real)

Como executar:
    pip install streamlit scikit-learn
    streamlit run app.py
"""

import re
from datetime import datetime

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. BASE DE CONHECIMENTO — INTENÇÕES E EXEMPLOS DE FRASES (NLU)
# =============================================================================
# Cada intenção tem frases de exemplo. O modelo de similaridade (TF-IDF +
# cosseno) compara a mensagem do usuário com essas frases para descobrir
# qual é a intenção mais provável — uma forma simples e transparente de
# "compreensão de linguagem natural", sem depender de serviços externos.

EXEMPLOS_INTENCAO = {
    "saudacao": [
        "oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí", "opa",
        "tudo bem", "oi tudo bem",
    ],
    "despedida": [
        "tchau", "até logo", "obrigado", "valeu", "muito obrigado",
        "pode encerrar", "até mais", "flw",
    ],
    "faq_conta_corrente": [
        "o que é conta corrente", "como funciona a conta corrente",
        "quero saber sobre conta corrente", "tem taxa na conta corrente",
    ],
    "faq_poupanca": [
        "o que é poupança", "como funciona a poupança",
        "vale a pena poupança", "rendimento da poupança",
    ],
    "faq_cartao_credito": [
        "o que é cartão de crédito", "como funciona o cartão de crédito",
        "limite do cartão", "fatura do cartão", "posso pagar só o mínimo do cartão",
        "o que é rotativo do cartão",
    ],
    "faq_emprestimo": [
        "o que é empréstimo pessoal", "como funciona um empréstimo",
        "quero saber sobre empréstimo", "juros do empréstimo",
    ],
    "faq_investimento": [
        "o que é CDB", "como funciona tesouro direto",
        "onde posso investir", "quero saber sobre investimentos",
        "vale a pena investir",
    ],
    "faq_pix": [
        "o que é pix", "como funciona o pix", "pix tem taxa",
        "é seguro usar pix",
    ],
    "calculo_juros_compostos": [
        "quero simular um investimento", "simular juros compostos",
        "quanto meu dinheiro rende", "simulação de rendimento",
        "quero calcular quanto meu investimento vai render",
    ],
    "calculo_financiamento": [
        "quero simular um financiamento", "simular parcelas de um empréstimo",
        "quanto fica a parcela", "calcular financiamento",
        "simular empréstimo",
    ],
    "ajuda": [
        "o que você faz", "como você pode me ajudar", "quais opções eu tenho",
        "menu", "ajuda",
    ],
}

EXPLICACOES_PRODUTOS = {
    "faq_conta_corrente": (
        "**Conta Corrente** é a conta usada no dia a dia para receber salário, "
        "pagar contas, fazer Pix e usar o cartão. Normalmente pode ter uma "
        "tarifa mensal de manutenção, que varia conforme o pacote de serviços "
        "contratado."
    ),
    "faq_poupanca": (
        "A **Poupança** é uma aplicação simples e de baixo risco. Ela rende "
        "mensalmente e tem liquidez imediata (você pode sacar quando quiser), "
        "mas historicamente rende menos que outras opções de investimento, "
        "como CDB ou Tesouro Direto."
    ),
    "faq_cartao_credito": (
        "O **Cartão de Crédito** permite comprar agora e pagar depois, dentro "
        "de um limite. Atenção: pagar apenas o **valor mínimo da fatura** faz "
        "o restante entrar no **rotativo**, que tem uma das taxas de juros "
        "mais altas do mercado. O ideal é sempre pagar a fatura integral."
    ),
    "faq_emprestimo": (
        "O **Empréstimo Pessoal** é uma quantia liberada pelo banco que você "
        "paga de volta em parcelas, com juros. As taxas variam conforme seu "
        "perfil de crédito, prazo e valor solicitado."
    ),
    "faq_investimento": (
        "Existem várias opções de investimento além da poupança, como "
        "**CDB** (empresta dinheiro ao banco e recebe juros) e **Tesouro "
        "Direto** (empresta dinheiro ao governo). Em geral, quanto maior o "
        "potencial de retorno, maior o risco — por isso é importante entender "
        "seu perfil antes de investir."
    ),
    "faq_pix": (
        "O **Pix** é o meio de pagamento instantâneo do Banco Central. "
        "Funciona 24 horas por dia, todos os dias, geralmente sem tarifa para "
        "pessoas físicas, e o dinheiro cai na conta do destinatário em "
        "segundos."
    ),
}

AVISO_SEGURANCA = (
    "\n\n_⚠️ Esta é uma simulação educativa e não substitui a orientação de "
    "um especialista financeiro qualificado._"
)


# =============================================================================
# 2. MOTOR DE RECONHECIMENTO DE INTENÇÃO (NLU simples via TF-IDF + cosseno)
# =============================================================================

@st.cache_resource
def montar_classificador():
    """Prepara o vetorizador TF-IDF treinado com as frases de exemplo."""
    frases, rotulos = [], []
    for intencao, exemplos in EXEMPLOS_INTENCAO.items():
        for frase in exemplos:
            frases.append(frase)
            rotulos.append(intencao)

    vetorizador = TfidfVectorizer()
    matriz = vetorizador.fit_transform(frases)
    return vetorizador, matriz, rotulos


def reconhecer_intencao(mensagem: str, limiar: float = 0.30) -> str:
    """Retorna a intenção mais provável para a mensagem, ou 'fallback'."""
    vetorizador, matriz, rotulos = montar_classificador()
    vetor_msg = vetorizador.transform([mensagem.lower()])
    similaridades = cosine_similarity(vetor_msg, matriz)[0]

    indice_melhor = similaridades.argmax()
    if similaridades[indice_melhor] < limiar:
        return "fallback"
    return rotulos[indice_melhor]


# =============================================================================
# 3. FUNÇÕES DE CÁLCULO FINANCEIRO (simulações demonstrativas)
# =============================================================================

def simular_juros_compostos(valor_inicial: float, taxa_mensal: float, meses: int) -> dict:
    """Calcula o montante final de um investimento com juros compostos."""
    montante = valor_inicial * (1 + taxa_mensal / 100) ** meses
    rendimento = montante - valor_inicial
    return {"montante": montante, "rendimento": rendimento}


def simular_financiamento(valor: float, taxa_mensal: float, parcelas: int) -> dict:
    """Calcula o valor da parcela fixa de um financiamento (Tabela Price)."""
    i = taxa_mensal / 100
    if i == 0:
        parcela = valor / parcelas
    else:
        parcela = valor * (i * (1 + i) ** parcelas) / ((1 + i) ** parcelas - 1)
    total_pago = parcela * parcelas
    juros_totais = total_pago - valor
    return {"parcela": parcela, "total_pago": total_pago, "juros_totais": juros_totais}


def extrair_numeros(texto: str) -> list:
    """Extrai números (inteiros ou decimais) de uma mensagem do usuário."""
    texto = texto.replace(",", ".")
    return [float(n) for n in re.findall(r"\d+\.?\d*", texto)]


# =============================================================================
# 4. ESTADO DA CONVERSA (persistência de contexto)
# =============================================================================

def iniciar_estado():
    """Inicializa as variáveis de sessão na primeira execução."""
    if "historico" not in st.session_state:
        st.session_state.historico = [
            {
                "papel": "assistente",
                "texto": (
                    "Olá! 👋 Eu sou o seu assistente financeiro virtual. "
                    "Posso explicar produtos do banco (conta, cartão, "
                    "empréstimo, investimentos, Pix) e fazer simulações "
                    "financeiras simples. Qual é o seu nome?"
                ),
            }
        ]
    if "contexto" not in st.session_state:
        st.session_state.contexto = {"nome_usuario": None, "aguardando_nome": True}
    if "calculo_pendente" not in st.session_state:
        st.session_state.calculo_pendente = None  # dict com tipo e parâmetros parciais


def registrar_mensagem(papel: str, texto: str):
    st.session_state.historico.append({"papel": papel, "texto": texto})


# =============================================================================
# 5. LÓGICA DE DIÁLOGO — ligando intenção + contexto à resposta
# =============================================================================

def tratar_extracao_de_nome(mensagem: str) -> str | None:
    """Tenta identificar o nome do usuário em frases como 'meu nome é Ana'."""
    padrao = re.search(r"(?:meu nome é|me chamo|sou o|sou a|sou)\s+([A-Za-zÀ-ÿ]+)", mensagem, re.IGNORECASE)
    if padrao:
        return padrao.group(1).capitalize()
    # Se o bot perguntou o nome e o usuário respondeu só com uma palavra
    if st.session_state.contexto.get("aguardando_nome") and len(mensagem.strip().split()) <= 2:
        return mensagem.strip().split()[0].capitalize()
    return None


def continuar_calculo_pendente(mensagem: str) -> str:
    """Conduz uma simulação financeira em múltiplas etapas, guardando o que
    já foi informado (isso é a 'persistência de contexto' da conversa)."""
    calc = st.session_state.calculo_pendente
    numeros = extrair_numeros(mensagem)
    if numeros:
        campo = calc["campos_faltando"][0]
        calc["parametros"][campo] = numeros[0]
        calc["campos_faltando"].pop(0)

    if calc["campos_faltando"]:
        proximo = calc["campos_faltando"][0]
        return PERGUNTAS_CAMPO[proximo]

    # Todos os parâmetros foram coletados: calcula o resultado
    st.session_state.calculo_pendente = None
    p = calc["parametros"]

    if calc["tipo"] == "juros_compostos":
        r = simular_juros_compostos(p["valor_inicial"], p["taxa_mensal"], int(p["meses"]))
        return (
            f"📈 **Simulação de Investimento**\n\n"
            f"- Valor inicial: R$ {p['valor_inicial']:.2f}\n"
            f"- Taxa mensal: {p['taxa_mensal']:.2f}%\n"
            f"- Período: {int(p['meses'])} meses\n\n"
            f"**Montante final estimado: R$ {r['montante']:.2f}**\n"
            f"Rendimento no período: R$ {r['rendimento']:.2f}"
            f"{AVISO_SEGURANCA}"
        )
    else:  # financiamento
        r = simular_financiamento(p["valor"], p["taxa_mensal"], int(p["parcelas"]))
        return (
            f"🧾 **Simulação de Financiamento**\n\n"
            f"- Valor financiado: R$ {p['valor']:.2f}\n"
            f"- Taxa mensal: {p['taxa_mensal']:.2f}%\n"
            f"- Número de parcelas: {int(p['parcelas'])}\n\n"
            f"**Parcela estimada: R$ {r['parcela']:.2f}**\n"
            f"Total pago ao final: R$ {r['total_pago']:.2f}\n"
            f"Total de juros: R$ {r['juros_totais']:.2f}"
            f"{AVISO_SEGURANCA}"
        )


PERGUNTAS_CAMPO = {
    "valor_inicial": "Qual o valor inicial que você pretende investir? (ex: 1000)",
    "taxa_mensal": "Qual a taxa de juros mensal estimada, em %? (ex: 1.2)",
    "meses": "Por quantos meses o dinheiro ficará investido?",
    "valor": "Qual o valor total do financiamento?",
    "parcelas": "Em quantas parcelas você pretende pagar?",
}


def iniciar_calculo(tipo: str, mensagem: str) -> str:
    """Inicia uma nova simulação, aproveitando números já citados na frase."""
    if tipo == "juros_compostos":
        campos = ["valor_inicial", "taxa_mensal", "meses"]
    else:
        campos = ["valor", "taxa_mensal", "parcelas"]

    st.session_state.calculo_pendente = {
        "tipo": tipo,
        "parametros": {},
        "campos_faltando": campos,
    }
    return continuar_calculo_pendente(mensagem)


def gerar_resposta(mensagem: str) -> str:
    """Função central: decide o que responder com base na intenção e no
    contexto acumulado da conversa."""
    contexto = st.session_state.contexto

    # Etapa 1: se ainda não sabemos o nome do usuário, tenta capturá-lo
    if contexto.get("aguardando_nome"):
        nome = tratar_extracao_de_nome(mensagem)
        if nome:
            contexto["nome_usuario"] = nome
            contexto["aguardando_nome"] = False
            return (
                f"Prazer em falar com você, **{nome}**! 😊 Posso te ajudar "
                f"explicando produtos financeiros (conta, cartão de crédito, "
                f"empréstimo, investimentos, Pix) ou fazendo uma simulação "
                f"de investimento/financiamento. O que você gostaria de saber?"
            )

    # Etapa 2: se há uma simulação em andamento, continua ela
    if st.session_state.calculo_pendente:
        return continuar_calculo_pendente(mensagem)

    # Etapa 3: reconhece a intenção da mensagem
    intencao = reconhecer_intencao(mensagem)
    nome = contexto.get("nome_usuario")
    saudacao_pessoal = f", {nome}" if nome else ""

    if intencao == "saudacao":
        return f"Olá novamente{saudacao_pessoal}! Em que posso ajudar hoje?"

    if intencao == "despedida":
        return f"Foi um prazer ajudar{saudacao_pessoal}! Se precisar de algo mais, é só chamar. 👋"

    if intencao == "ajuda":
        return (
            "Posso ajudar você com:\n"
            "- 📘 Explicações sobre **conta corrente, poupança, cartão de crédito, "
            "empréstimo, investimentos e Pix**\n"
            "- 📈 **Simulação de investimento** (juros compostos)\n"
            "- 🧾 **Simulação de financiamento/empréstimo**\n\n"
            "É só me perguntar, por exemplo: *'quero simular um investimento'* "
            "ou *'o que é cartão de crédito'*."
        )

    if intencao in EXPLICACOES_PRODUTOS:
        return EXPLICACOES_PRODUTOS[intencao] + f"\n\nPosso explicar outro produto ou fazer uma simulação, {nome or 'se quiser'}."

    if intencao == "calculo_juros_compostos":
        return iniciar_calculo("juros_compostos", mensagem)

    if intencao == "calculo_financiamento":
        return iniciar_calculo("financiamento", mensagem)

    # Fallback: não entendeu a intenção
    return (
        "Desculpe, não entendi muito bem 🤔. Posso explicar produtos "
        "financeiros (conta, cartão, empréstimo, investimentos, Pix) ou "
        "fazer uma simulação de investimento ou financiamento. Pode "
        "reformular sua pergunta?"
    )


# =============================================================================
# 6. INTERFACE (Streamlit) — princípios de UX aplicados
# =============================================================================
# - Clareza: respostas objetivas, em blocos curtos, com destaque visual (markdown)
# - Segurança: avisos explícitos em toda simulação financeira
# - Personalização: uso do nome do usuário e do histórico da conversa
# - Transparência: barra lateral explica o que o assistente sabe fazer

st.set_page_config(page_title="Assistente Financeiro IA", page_icon="💬", layout="centered")

iniciar_estado()

with st.sidebar:
    st.header("💬 Assistente Financeiro IA")
    st.write(
        "Experiência digital de relacionamento financeiro guiada por IA, "
        "com FAQ inteligente, simulações financeiras e memória de contexto."
    )
    st.markdown("**O que você pode perguntar:**")
    st.markdown(
        "- O que é cartão de crédito / Pix / poupança?\n"
        "- Quero simular um investimento\n"
        "- Quero simular um financiamento"
    )
    st.divider()
    if st.session_state.contexto.get("nome_usuario"):
        st.success(f"Conversando com: {st.session_state.contexto['nome_usuario']}")
    st.caption(f"Sessão iniciada em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if st.button("🔄 Reiniciar conversa"):
        st.session_state.clear()
        st.rerun()

st.title("💬 Assistente Financeiro IA")
st.caption("Converse comigo sobre produtos financeiros ou peça uma simulação.")

# Exibe o histórico da conversa
for item in st.session_state.historico:
    with st.chat_message("assistant" if item["papel"] == "assistente" else "user"):
        st.markdown(item["texto"])

# Campo de entrada do usuário
entrada_usuario = st.chat_input("Digite sua mensagem...")

if entrada_usuario:
    registrar_mensagem("usuario", entrada_usuario)
    with st.chat_message("user"):
        st.markdown(entrada_usuario)

    resposta = gerar_resposta(entrada_usuario)
    registrar_mensagem("assistente", resposta)
    with st.chat_message("assistant"):
        st.markdown(resposta)
