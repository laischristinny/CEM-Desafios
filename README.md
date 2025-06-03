# 🧲 App de Transformadores - Desafios

Este é um aplicativo interativo construído com **Streamlit** para auxiliar no **dimensionamento, análise e visualização 3D** de transformadores monofásicos. Ele está dividido em quatro desafios principais e é voltado tanto para fins educacionais quanto para aplicações técnicas.

## 📦 Funcionalidades

### 🔹 Desafio 1 - Dimensionamento
- Entrada de parâmetros: tensões primárias/secundárias, potência, frequência e tipo de lâmina.
- Cálculo de:
  - Corrente e seção dos condutores
  - Número de espiras do primário e secundário
  - Seção magnética e geométrica do núcleo
  - Seleção de lâmina
  - Peso estimado de ferro e cobre
  - Verificação de viabilidade (relação Sj/Scu)
- Visualização 3D do núcleo e espiras (via Plotly)

### 🔹 Desafio 2 - Magnetização
- Análise do fluxo magnético em função do tempo
- Leitura de curva de magnetização a partir de planilha Excel (`MagCurve.xlsx`)
- Interpolação da curva B-H para simulação
- Cálculo de corrente de magnetização com base na curva do material

### 🔹 Desafio 3 - Calculadora de Parâmetros de Transformador Monofásico
- Cálculo dos parâmetros:
  - Resistência do núcleo (Rc)
  - Reatância de magnetização (Xm)
  - Corrente ativa (Ic) e reativa (Im) no circuito aberto
  - Resistência equivalente (Req)
  - Reatância equivalente (Xeq)
- Opção de visualização dos parâmetros no modelo equivalente:
  - Modelo em Série
  - Modelo em T
  - Modelo em L
- Conversão dos parâmetros referidos ao primário ou secundário.
- Geração de Diagrama Fasorial da Corrente de Excitação.

### 🔹 Desafio 4 - Regulação de Tensão
Este módulo calcula a regulação de tensão de um transformador considerando seus parâmetros equivalentes (resistência e reatância) e o fator de potência da carga. Além disso, gera um diagrama fasorial interativo representando as tensões e quedas internas no transformador.
Cálculo da regulação de tensão aproximada em percentual (%).

- Visualização do diagrama fasorial com os vetores:
- Tensão no secundário (V₂)
- Queda resistiva (I·Rₑq)
- Queda reativa (j·I·Xₑq)
- Tensão no primário estimada (V₁ aprox.)

---

## 🚀 Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/laischristinny/CEM-Desafios.git
cd CEM-Desafios
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o aplicativo
```bash
streamlit run app.py
```

## 💻 Ou... visualize e interaja com o app pela nuvem
Acesse: https://cem-desafios.streamlit.app/
