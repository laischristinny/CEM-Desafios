import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


st.set_page_config(layout="wide")
st.title("App de Transformadores - Desafios")

tab1, tab2, tab3, tab4 = st.tabs(["Desafio 1 - Dimensionamento", "Desafio 2 - Magnetização", "Desafio 3 - Transformador Monofásico", "Desafio 4 - Regulação de Tensão"])

with tab1:
    # ======= ENTRADAS =======

    st.header("Dados de entrada")

    tipo_transformador = st.selectbox(
        "Tipo de Transformador",
        [
            "Transformador de um primário e um secundário",
            "Transformador de dois primários e um secundário ou um primário e dois secundários",
            "Transformador de dois primários e dois secundários"
        ]
    )
    V1_input = st.text_input("Tensão Primária (V1) em V (use / para múltiplos)", value="120")
    V2_input = st.text_input("Tensão Secundária (V2) em V (use / para múltiplos)", value="220")

    V1_list = [int(v.strip()) for v in V1_input.split("/") if v.strip().isdigit()]
    V2_list = [int(v.strip()) for v in V2_input.split("/") if v.strip().isdigit()]
    W2 = st.number_input("Potência (W2) em VA", value=300)
    f = st.number_input("Frequência em Hz", value=50)
    tipo_de_lamina = st.selectbox("Tipo de Lâmina do Núcleo", ["Padronizada", "Comprida"])

    # ======= CÁLCULOS =======

    W1 = 1.1 * W2  # Potência primária
    I1 = [round(W1 / v, 2) for v in V1_list] # Corrente primária
    d = 3 if W2 <= 500 else 2.5 if W2 <= 1000 else 2 # Densidade da corrente
    S1 = [round(i / d, 2) for i in I1] # Seção do condutor primário

    I2 = [round(W2 / v, 2) for v in V2_list] # Corrente secundária
    S2 = [round(i / d, 2) for i in I2] # Seção do condutor secundário

    awg_table = [
        {"AWG": 25, "area_mm2": 0.162},
        {"AWG": 24, "area_mm2": 0.205},
        {"AWG": 23, "area_mm2": 0.258},
        {"AWG": 22, "area_mm2": 0.326},
        {"AWG": 21, "area_mm2": 0.410},
        {"AWG": 20, "area_mm2": 0.518},
        {"AWG": 19, "area_mm2": 0.653},
        {"AWG": 18, "area_mm2": 0.823},
        {"AWG": 17, "area_mm2": 1.04},
        {"AWG": 16, "area_mm2": 1.31},
        {"AWG": 15, "area_mm2": 1.65},
        {"AWG": 14, "area_mm2": 2.08},
        {"AWG": 13, "area_mm2": 2.62},
        {"AWG": 12, "area_mm2": 3.31},
        {"AWG": 11, "area_mm2": 4.17},
        {"AWG": 10, "area_mm2": 5.26},
        {"AWG": 9,  "area_mm2": 6.63},
        {"AWG": 8,  "area_mm2": 8.37},
        {"AWG": 7,  "area_mm2": 10.55},
        {"AWG": 6,  "area_mm2": 13.30},
        {"AWG": 5,  "area_mm2": 16.80},
        {"AWG": 4,  "area_mm2": 21.15},
        {"AWG": 3,  "area_mm2": 26.67},
        {"AWG": 2,  "area_mm2": 33.62},
        {"AWG": 1,  "area_mm2": 42.41},
        {"AWG": 0,  "area_mm2": 53.49},
    ]

    def encontrar_awg_por_secao(secao_mm2):
        for fio in awg_table:
            if fio["area_mm2"] >= secao_mm2:
                return fio
        return None

    fio_awg_s1_1 = encontrar_awg_por_secao(S1[0])
    fio_awg_s2_1 = encontrar_awg_por_secao(S2[0])
    if len(S1) >= 2:
        fio_awg_s1_2 = encontrar_awg_por_secao(S1[1])
    if len(S2) >= 2:
        fio_awg_s2_2 = encontrar_awg_por_secao(S2[1])

    fator_tipo = {
        "Transformador de um primário e um secundário": 1,
        "Transformador de dois primários e um secundário ou um primário e dois secundários": 1.25,
        "Transformador de dois primários e dois secundários": 1.5
    }[tipo_transformador]

    coef = 7.5 if tipo_de_lamina == "Padronizada" else 6
    Sm = round(coef * math.sqrt((fator_tipo * W2) / f), 1) # Seção magnética do núcleo
    Sg = round(Sm * 1.1, 1)  # Seção geométrica do núcleo

    a = math.ceil(math.sqrt(Sg)) # Largura da coluna central  do transformador
    b = round(Sg / a)  # Comprimento do pacote laminado

    # Função para seleção de lâminas
    def selecionar_lamina(a, tipo):
        laminas_padronizadas = [
            {"numero": 0, "a_cm": 1.5, "secao_mm2": 168, "peso_kgcm": 0.095},
            {"numero": 1, "a_cm": 2, "secao_mm2": 300, "peso_kgcm": 0.170},
            {"numero": 2, "a_cm": 2.5, "secao_mm2": 468, "peso_kgcm": 0.273},
            {"numero": 3, "a_cm": 3, "secao_mm2": 675, "peso_kgcm": 0.380},
            {"numero": 4, "a_cm": 3.5, "secao_mm2": 900, "peso_kgcm": 0.516},
            {"numero": 5, "a_cm": 4, "secao_mm2": 1200, "peso_kgcm": 0.674},
            {"numero": 6, "a_cm": 5, "secao_mm2": 1880, "peso_kgcm": 1.053}
        ]
        laminas_compridas = [
            {"numero": 5, "a_cm": 4, "secao_mm2": 2400, "peso_kgcm": 1.000},
            {"numero": 6, "a_cm": 5, "secao_mm2": 3750, "peso_kgcm": 1.580}
        ]
        laminas = laminas_padronizadas if tipo == "Padronizada" else laminas_compridas
        for lamina in laminas:
            if a <= lamina["a_cm"]:
                return lamina
        return laminas[-1]

    numero_lamina = selecionar_lamina(a, tipo_de_lamina)

    # Dimensões efetivas do nucleo central
    SgEfetivo = a * b
    SmEfetivo = round(SgEfetivo / 1.1, 2)

    if f == 50:
        EspVolt = round(40 / SmEfetivo, 2)
    elif f == 60:
        EspVolt = round(33.5 / SmEfetivo, 2)
    else:
        EspVolt = round((1e8 / (4.44 * 11300 * f)) / SmEfetivo, 2)

    N1 = math.ceil(EspVolt * V1_list[0])  # Espiras do primário
    N2 = math.ceil(EspVolt * V2_list[0] * 1.1) # Espiras do secundário

    if len(S1) >= 2 and len(S2) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N1 * fio_awg_s1_2['area_mm2'] + N2 * fio_awg_s2_1['area_mm2'] + N2 * fio_awg_s2_2['area_mm2']
    elif len(S1) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N1 * fio_awg_s1_2['area_mm2'] + N2 * fio_awg_s2_1['area_mm2']
    elif len(S2) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N2 * fio_awg_s2_1['area_mm2'] + N2 * fio_awg_s2_2['area_mm2']
    else:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N2 * fio_awg_s2_1['area_mm2']

    Sj = numero_lamina["secao_mm2"] # Seção da janela
    executavel = Sj / Scu

    Pfe = numero_lamina["peso_kgcm"] * b # Peso do ferro
    lm = (2 * a) + (2 * b) + (0.5 * a * math.pi)  # Comprimento da espira média do cobre
    Pcu = (Scu / 100 * lm * 9) / 1000 # Peso do cobre

    # ======= RESULTADOS =======

    st.header("Resultados do Dimensionamento")

    st.subheader("Número de Espiras")
    st.write(f"Primário: {N1} espiras")
    st.write(f"Secundário: {N2} espiras")

    st.subheader("Bitola dos Cabos")
    st.write(f"Tipo de fio AWG para S1 Primário 1: {fio_awg_s1_1['AWG']}, cuja seção é  {fio_awg_s1_1['area_mm2']}")
    st.write(f"Tipo de fio AWG para S2 Secundário 1: {fio_awg_s2_1['AWG']}, cuja seção é  {fio_awg_s2_1['area_mm2']}")
    if len(S1) >= 2:
        st.write(f"Tipo de fio AWG para S1 Primário 2: {fio_awg_s1_2['AWG']}, cuja seção é  {fio_awg_s1_2['area_mm2']}")
    if len(S2) >= 2:
        st.write(f"Tipo de fio AWG para S2 Secundário 2: {fio_awg_s2_2['AWG']}, cuja seção é  {fio_awg_s2_2['area_mm2']}")

    for i, (v, s) in enumerate(zip(V1_list, S1)):
        st.write(f"Primário {i+1} ({v} V): {s:.2f} mm²")
    for i, (v, s) in enumerate(zip(V2_list, S2)):
        st.write(f"Secundário {i+1} ({v} V): {s:.2f} mm²")

    st.subheader("Lâmina do Núcleo")
    st.write(f"Tipo de lâmina selecionada: Nº {numero_lamina['numero']} com seção {numero_lamina['secao_mm2']} mm²")
    st.write(f"Quantidade de lâminas aproximada (baseado no comprimento b): {b} unidades")

    st.subheader("Dimensões do Transformador")
    st.write(f"Largura da coluna central (a): {a} cm")
    st.write(f"Comprimento do pacote laminado (b): {b} cm")
    st.write(f"Seção geométrica efetiva do núcleo (SgEfetivo): {SgEfetivo:.2f} cm²")
    st.write(f"Seção magnética efetiva do núcleo (SmEfetivo): {SmEfetivo:.2f} cm²")

    st.subheader("Peso do Transformador")
    st.write(f"Peso do núcleo de ferro (Pfe): {Pfe:.2f} kg")
    st.write(f"Peso estimado do cobre (Pcu): {Pcu:.2f} kg")

    if executavel >= 3:
        st.success("Transformador é executável conforme critério de relação Sj/Scu >= 3.")
    else:
        st.warning("Transformador não executável.")

    # ======= VISUALIZAÇÃO 3D =======

    def create_transformer_sections(x, y, z, dx, dy, dz):
        return np.array([
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ])

    def rotate_transformer(vertices, angle):
        rotation_matrix = np.array([
            [np.cos(angle), 1, 0],
            [0, -np.cos(angle), 1],
            [np.sin(angle), np.cos(angle), 0]
        ])
        return np.dot(vertices, rotation_matrix.T)

    def plot_box(fig, vertices, color='gray', opacity=0.5):
        faces = [
            [0, 1, 5, 4], [7, 6, 2, 3],
            [0, 3, 7, 4], [1, 2, 6, 5],
            [0, 1, 2, 3], [4, 5, 6, 7]
        ]

        for face in faces:
            x = [vertices[i][0] for i in face] + [vertices[face[0]][0]]
            y = [vertices[i][1] for i in face] + [vertices[face[0]][1]]
            z = [vertices[i][2] for i in face] + [vertices[face[0]][2]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            ))

    def add_coils(fig, a, b, V1, V2):
        offset_factor = 0.5 * a

        max_voltage = max(V1 + V2)

        for i, v1 in enumerate(V1):
            n_points = int(100 * (v1 / max_voltage))  # proporcional à maior tensão
            z = np.linspace(a, 0.5*a, n_points) + a*1.01
            x = (b/1.5)*np.cos(z*3*b) - (a*(-1.52)) - i * offset_factor
            y = np.sin(z*3*b)*(b/1.5)+(b/2)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='brown', width=4),
                name=f'Primário {i+1}'
            ))

        for j, v2 in enumerate(V2):
            n_points = int(100 * (v2 / max_voltage))  # proporcional à maior tensão
            z = np.linspace(0.6*a, a*1.2, n_points) - a*0.01
            x = (b/1.5)*(np.cos(z*2*a)-2.5) + (b*2.79) + j * offset_factor
            y = np.sin(z*2*a)*(b/1.5)+(b/2)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='yellow', width=4),
                name=f'Secundário {j+1}'
            ))

    def generate_transformer_plot(angle, a, b, V1, V2):
        fig = go.Figure()
        parts = [
            create_transformer_sections(0, 0, 0, 0.5*a, 3*a, b),
            create_transformer_sections(0, a+1.5*a, 0, 2*a, a*0.5, b),
            create_transformer_sections(0, a, 0, 2*a, a, b),
            create_transformer_sections(0, 0, 0, 2*a, a*0.5, b),
            create_transformer_sections(2*a, 0, 0, 0.5*a, 3*a, b)
        ]

        for part in parts:
            rotated = rotate_transformer(part, angle)
            plot_box(fig, rotated, color='gray', opacity=0.4)

        add_coils(fig, a, b, V1, V2)

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=750,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        return fig

    st.title("Transformador Monofásico - Visualização 3D Interativa")

    if st.button("Gerar Transformador"):
        angle = np.radians(90)

        # Converter strings para floats
        V1 = [float(v) for v in V1_list]
        V2 = [float(v) for v in V2_list]
        print(f"v1 é {V1} e v2 é {V2}")

        fig = generate_transformer_plot(angle, a, b, V1, V2)
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Desafio 2")
    VM = st.number_input("Tensão Máxima", value=325)
    N = st.number_input("Número de Espiras", value=850)
    freq = st.number_input("Frequência (Hz)", value=50)

    try:
        # Lê o Excel diretamente do diretório atual
        df = pd.read_excel("MagCurve.xlsx")

        # Verifica se as colunas estão corretas
        if 'MMF' in df.columns and 'Fluxo' in df.columns:
            fmm_data = df['MMF'].values
            fluxo_data = df['Fluxo'].values

            from scipy.interpolate import interp1d

            # Interpolação linear
            fluxo_para_fmm = interp1d(fluxo_data, fmm_data, kind='linear', fill_value="extrapolate")

            # Parâmetros elétricos
            w = 2 * np.pi * freq

            # Tempo de simulação (0 a 340ms) com passo de 1/3000 s
            t = np.arange(0, 0.340, 1/3000)  # Vai até 340 ms

            # Fluxo magnético a partir da tensão (mesmo cálculo do MATLAB)
            fluxo_t = -VM / (w * N) * np.cos(w * t)

            # FMM correspondente ao fluxo, via interpolação
            fmm_t = fluxo_para_fmm(fluxo_t)

            # Corrente de magnetização
            corrente_t = fmm_t / N

            # Plotar
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t * 1000, corrente_t, color='blue')
            ax.set_xlabel('Tempo (ms)')
            ax.set_ylabel('Corrente de Magnetização (A)')
            ax.set_title('Corrente de Magnetização x Tempo')
            ax.grid(True)

            st.pyplot(fig)
        else:
            st.error("A planilha 'MagCurve.xlsx' deve conter as colunas 'MMF' e 'Fluxo'.")

    except FileNotFoundError:
        st.error("Arquivo 'MagCurve.xlsx' não encontrado no diretório atual.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler a planilha: {e}")

with tab3:
    st.subheader("Ensaio em Circuito Aberto")
    Vca = st.number_input("Tensão (Vca) [V]", min_value=0.0, value=220.0)
    Ica = st.number_input("Corrente (Ica) [A]", min_value=0.0, value=1.0)
    Pca = st.number_input("Potência (Pca) [W]", min_value=0.0, value=100.0)

    # Entradas do ensaio em curto-circuito
    st.subheader("Ensaio em Curto-Circuito")
    Vcc = st.number_input("Tensão (Vcc) [V]", min_value=0.0, value=50.0)
    Icc = st.number_input("Corrente (Icc) [A]", min_value=0.0, value=5.0)
    Pcc = st.number_input("Potência (Pcc) [W]", min_value=0.0, value=200.0)

    if st.button("Calcular Parâmetros"):
        st.markdown("## Resultados")

        # Circuito aberto - cálculo de Rc e Xm
        cos_phi0 = Pca / (Vca * Ica)
        phi0 = np.arccos(cos_phi0)
        Iw = Ica * cos_phi0
        Im = Ica * np.sin(phi0)

        Rc = Vca / Iw if Iw != 0 else float('inf')
        Xm = Vca / Im if Im != 0 else float('inf')

        # Curto-circuito - cálculo de Re (resistência equivalente) e Xe (reatância equivalente)
        cos_phicc = Pcc / (Vcc * Icc)
        phicc = np.arccos(cos_phicc)
        Zeq = Vcc / Icc
        Re = Zeq * cos_phicc
        Xe = Zeq * np.sin(phicc)

        st.write(f"**Rc (resistência de núcleo):** {Rc:.2f} Ω")
        st.write(f"**Xm (reatância de magnetização):** {Xm:.2f} Ω")
        st.write(f"**Re (resistência equivalente):** {Re:.2f} Ω")
        st.write(f"**Xe (reatância equivalente):** {Xe:.2f} Ω")

        # Diagrama Fasorial
        st.markdown("### Diagrama Fasorial")
        fig, ax = plt.subplots()

        # Tensão como vetor referência (horizontal)
        V_mod = Vca / max(Vca, Ica)  # Normaliza para ficar proporcional aos vetores de corrente
        # Normaliza os vetores
        max_magnitude = max(abs(V_mod), abs(Iw), abs(Im), abs(Ica))
        V_mod_normalizado = V_mod / max_magnitude
        Iw_normalizado = Iw / max_magnitude
        Im_normalizado = Im / max_magnitude
        Ica_normalizado = Ica / max_magnitude

        # Calcular os valores máximos absolutos dos vetores
        max_val = max(abs(V_mod), abs(Iw), abs(Im), abs(Ica))

        # Definir limites para os eixos com base no valor máximo
        # Vamos definir um fator de ampliação para garantir que o gráfico não fique muito apertado
        fator_ampliacao = 1.1  # 10% a mais do que o valor máximo

        # Ajustar os limites dos eixos
        ax.set_xlim(-max_val * fator_ampliacao, max_val * fator_ampliacao)
        ax.set_ylim(-max_val * fator_ampliacao, max_val * fator_ampliacao)

        # Plota os vetores normalizados
        ax.quiver(0, 0, V_mod_normalizado, 0, angles='xy', scale_units='xy', scale=1, color='orange', label='Vca (referência)')
        ax.quiver(0, 0, Iw_normalizado, 0, angles='xy', scale_units='xy', scale=1, color='r', label='Iw (A)')
        ax.quiver(0, 0, 0, Im_normalizado, angles='xy', scale_units='xy', scale=1, color='b', label='Im (A)')
        ax.quiver(0, 0, Iw_normalizado, Im_normalizado, angles='xy', scale_units='xy', scale=1, color='g', label='Ica (A)')

        # Adiciona o ângulo phi0
        cos_phi0 = Pca / (Vca * Ica)
        if abs(cos_phi0) > 1:
            st.write(f"Valor inválido: Pca = {Pca} excede Vca*Ica = {Vca*Ica}")
        else:
            cos_phi0 = np.clip(cos_phi0, -1.0, 1.0)
            phi0 = np.arccos(cos_phi0)
            phi0_deg = np.degrees(phi0)

        arc = Arc((0, 0), 0.5, 0.5, angle=0, theta1=0, theta2=phi0_deg, color='gray', linestyle='--')
        ax.add_patch(arc)
        ax.text(0.35, 0.05, f'φ₀ = {phi0_deg:.2f}°', fontsize=10, color='gray')

        # Título, rótulos e grid
        ax.set_xlabel("Eixo Real")
        ax.set_ylabel("Eixo Imaginário")
        ax.grid(True)
        # Coloca a legenda fora do gráfico
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title("Diagrama Fasorial (Ensaio em Circuito Aberto)")

        # Exibe o gráfico
        st.pyplot(fig)

    #     # Ilustração do circuito equivalente
    #     st.markdown("### Circuito Equivalente (Modelo Simplificado)")

    #     st.image(
    #     "https://www.researchgate.net/profile/Josnier-Ramos-Guardarrama/publication/331683668/figure/fig4/AS:735195284671488@1557984930854/Figura-19-Circuito-equivalente-de-un-transformador-monofasico.png",
    #     caption="Fonte: Ramos-Guardarrama, ResearchGate (uso educacional)",
    #     use_container_width=True
    # )

with tab4:
    st.header("Desafio 4 - Regulação de Tensão")
    I2 = st.number_input("Corrente de carga I2 (A)", value=5.0)
    V2 = st.number_input("Tensão no secundário V2 (V)", value=220.0)
    R_eq = st.number_input("Resistência equivalente R_eq (Ω)", value=0.5)
    X_eq = st.number_input("Reatância equivalente X_eq (Ω)", value=1.2)
    cos_phi = st.number_input("Fator de potência cosφ", value=0.85, min_value=0.0, max_value=1.0)
    sin_phi = np.sqrt(1 - cos_phi**2)
    regulacao = (I2 * (R_eq * cos_phi + X_eq * sin_phi)) / V2 * 100
    st.write(f"Regulação estimada: **{regulacao:.2f}%**")
    V2_vec = np.array([V2, 0])
    VR = np.array([R_eq * I2 * cos_phi, R_eq * I2 * sin_phi])
    VX = np.array([-X_eq * I2 * sin_phi, X_eq * I2 * cos_phi])
    V1 = V2_vec + VR + VX
    fig = go.Figure()
    def arrow(name, vec, color):
        fig.add_trace(go.Scatter(x=[0, vec[0]], y=[0, vec[1]], mode="lines+text", name=name, line=dict(width=3, color=color), text=[None, name], textposition="top center"))
    arrow("V₂", V2_vec, "green")
    arrow("I·R_eq", VR, "blue")
    arrow("j·I·X_eq", VX, "orange")
    arrow("V₁ (aprox)", V1, "red")
    fig.update_layout(title="Diagrama Fasorial", xaxis_title="Eixo Real", yaxis_title="Eixo Imaginário", width=600, height=400, showlegend=True)
    st.plotly_chart(fig)
