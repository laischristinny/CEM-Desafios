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

### 🔹 Desafio 3 - Regulação e Rendimento
- Cálculo do rendimento do transformador com base em perdas no cobre e ferro
- Análise da **regulação de tensão** em função da carga
- Gráficos dinâmicos de rendimento e regulação para diferentes regimes de carga
- Possibilidade de simular cargas indutivas, resistivas ou capacitivas

### 🔹 Desafio 4 - Transformador com Tomadas
- Simulação de transformador com **tomadas de variação de tensão**
- Avaliação do efeito das tomadas sobre a tensão secundária
- Visualização gráfica das alterações de tensão conforme diferentes posições da tomada
- Útil para estudo de transformadores com múltiplas tensões de saída

---

## 🚀 Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/CEM-Desafios.git
cd Desafio1CEM
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
