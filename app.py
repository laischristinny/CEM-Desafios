import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

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
    a = numero_lamina["a_cm"]
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

    def criar_seções_do_transformador(x, y, z, dx, dy, dz):
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

    def rotacionar_transformador(vertices, angle):
        rotation_matrix = np.array([
            [np.cos(angle), 1, 0],
            [0, -np.cos(angle), 1],
            [np.sin(angle), np.cos(angle), 0]
        ])
        return np.dot(vertices, rotation_matrix.T)

    def plot_transformador(fig, vertices, color='gray', opacity=0.5):
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

    def add_espiras(fig, a, b, V1_voltages, V2_voltages):
        common_x_center = 1.5 * a
        common_y_center = b / 2

        V1 = [float(v) for v in V1_voltages]
        V2 = [float(v) for v in V2_voltages]
        all_voltages = V1 + V2

        max_voltage = max([v for v in all_voltages if v > 0], default=1.0)

        z_center = 1.25 * a
        coil_height = 0.8 * a
        z_min = z_center - coil_height / 2
        z_max = z_center + coil_height / 2

        params = {
            'gap': b / 40,
            'sec_thickness': b / 7,
            'pri_thickness': b / 8,
            'outer_limit': b / 2.05,
            'min_pri': b / 30,
            'min_sec': b / 25
        }

        def calc_radii(num, outer, thickness, min_fallback):
            if num <= 0:
                return []
            if num == 1:
                return [outer * 0.98]
            inner = max(outer - thickness, min_fallback)
            if inner >= outer:
                inner = outer * 0.7
            return np.linspace(inner, outer, num).tolist()

        # Secundário
        radii_s = calc_radii(len(V2), params['outer_limit'], params['sec_thickness'], params['min_sec'])
        limit_pri = (radii_s[0] - params['gap']) if radii_s else (params['outer_limit'] - params['gap'])

        # Primário
        radii_p = calc_radii(len(V1), limit_pri, params['pri_thickness'], params['min_pri'])

        # Garante raios mínimos
        radii_p = [max(r, params['min_pri'] * 0.5) for r in radii_p]
        radii_s = [max(r, params['min_sec'] * 0.5) for r in radii_s]

        def draw_coils(voltages, radii, color, name_prefix, angle_factor):
            for i, v in enumerate(voltages):
                if i >= len(radii): continue
                radius = radii[i]
                relative_v = v / max_voltage

                turn_density = 0.5 + (1.5 - 0.5) * relative_v
                n_points = max(int(100 * relative_v), 10)

                z_vals = np.linspace(z_min, z_max, n_points)
                angle = z_vals * angle_factor * turn_density

                x = common_x_center + radius * np.cos(angle)
                y = common_y_center + radius * np.sin(angle)

                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z_vals,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'{name_prefix} {i+1}'
                ))

        draw_coils(V1, radii_p, 'brown', 'Primário', 3 * b)
        draw_coils(V2, radii_s, 'yellow', 'Secundário', 2 * a)

    def gerar_visu_transformador(angle, a, b, V1, V2):
        fig = go.Figure()
        parts = [
            criar_seções_do_transformador(0, 0, 0, 0.5*a, 3*a, b),
            criar_seções_do_transformador(0, a+1.5*a, 0, 2*a, a*0.5, b),
            criar_seções_do_transformador(0, a, 0, 2*a, a, b),
            criar_seções_do_transformador(0, 0, 0, 2*a, a*0.5, b),
            criar_seções_do_transformador(2*a, 0, 0, 0.5*a, 3*a, b)
        ]

        for part in parts:
            rotated = rotacionar_transformador(part, angle)
            plot_transformador(fig, rotated, color='gray', opacity=0.4)

        add_espiras(fig, a, b, V1, V2)

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

        fig = gerar_visu_transformador(angle, a, b, V1, V2)
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
    st.title("Transformador Monofásico - Calculadora de Parâmetros")

    # Entradas do transformador
    N1 = st.number_input("Número de espiras do primário (N1)", min_value=1, value=1000)
    N2 = st.number_input("Número de espiras do secundário (N2)", min_value=1, value=100)

    circuit_type = st.selectbox("Tipo de Circuito Equivalente", ["", "T", "L", "Série"])
    referred_to = st.selectbox("Referido a", ["", "primário", "secundário"])

    st.markdown("## Conjunto de Dados A")
    setA_type = st.selectbox("Tipo de Ensaio A", ["", "circuito-aberto", "curto-circuito"])
    setA_side = st.selectbox("Lado do Ensaio A", ["", "baixo", "alto"])
    Va = st.number_input("Tensão Va (V)", min_value=0.0, value=220.0)
    Ia = st.number_input("Corrente Ia (A)", min_value=0.0, value=1.0)
    Pa = st.number_input("Potência Pa (W)", min_value=0.0, value=100.0)

    st.markdown("## Conjunto de Dados B")
    setB_type = st.selectbox("Tipo de Ensaio B", ["", "circuito-aberto", "curto-circuito"])
    setB_side = st.selectbox("Lado do Ensaio B", ["", "baixo", "alto"])
    Vb = st.number_input("Tensão Vb (V)", min_value=0.0, value=50.0)
    Ib = st.number_input("Corrente Ib (A)", min_value=0.0, value=5.0)
    Pb = st.number_input("Potência Pb (W)", min_value=0.0, value=200.0)

    if st.button("Calcular Parâmetros"):
        try:
            if Ia == 0 or Ib == 0:
                raise ValueError("Corrente não pode ser zero.")
            if Pa == 0 or Pb == 0:
                raise ValueError("Potência não pode ser zero.")
            if Va == 0 or Vb == 0:
                raise ValueError("Tensão não pode ser zero.")

            a = N1 / N2
            a2 = a ** 2

            def calc_open(V, I, P):
                if I == 0 or P == 0:
                    raise ValueError("Corrente e potência devem ser maiores que zero para o circuito aberto.")
                Rc = (V**2) / P
                Zphi = V / I
                under_root = 1 / (Zphi**2) - 1 / (Rc**2)
                if under_root < 0:
                    raise ValueError("Impossível calcular Xm: raiz de número negativo.")
                Xm = 1 / np.sqrt(under_root)
                return Rc, Xm, Zphi

            def calc_short(V, I, P):
                if I == 0:
                    raise ValueError("Corrente deve ser maior que zero para o curto-circuito.")
                Zcc = V / I
                Req = P / (I**2)
                under_root = Zcc**2 - Req**2
                if under_root < 0:
                    raise ValueError("Impossível calcular Xeq: raiz de número negativo.")
                Xeq = np.sqrt(under_root)
                return Req, Xeq, Zcc

            Rc = Xm = Zphi = Req = Xeq = Rc_prime = Xm_prime = Iphi_prime = Ic = Im = 0

            if setA_type == "curto-circuito" and setA_side == "baixo" and setB_type == "circuito-aberto" and setB_side == "alto":
                Req, Xeq, _ = calc_short(Va, Ia, Pa)
                Rc, Xm, Zphi = calc_open(Vb, Ib, Pb)
                Rc_prime = Rc * a2
                Xm_prime = Xm * a2
                Iphi_prime = Ib / a
                Ic = Vb / Rc_prime
                Im = Vb / Xm_prime

            elif setA_type == "circuito-aberto" and setA_side == "baixo" and setB_type == "curto-circuito" and setB_side == "alto":
                Rc, Xm, Zphi = calc_open(Va, Ia, Pa)
                Rc_prime = Rc * a2
                Xm_prime = Xm * a2
                Iphi_prime = Ia / a
                Ic = Va / Rc_prime
                Im = Va / Xm_prime
                Req, Xeq, _ = calc_short(Vb, Ib, Pb)

            elif setA_type == "curto-circuito" and setA_side == "alto" and setB_type == "circuito-aberto" and setB_side == "baixo":
                Req, Xeq, _ = calc_short(Va, Ia, Pa)
                Rc, Xm, Zphi = calc_open(Vb, Ib, Pb)

            elif setA_type == "circuito-aberto" and setA_side == "alto" and setB_type == "curto-circuito" and setB_side == "baixo":
                Rc, Xm, Zphi = calc_open(Va, Ia, Pa)
                Rc_prime = Rc * a2
                Xm_prime = Xm * a2
                Iphi_prime = Ia * (N2 / N1)
                Ic = Va / Rc_prime
                Im = Va / Xm_prime
                Req, Xeq, _ = calc_short(Vb, Ib, Pb)
            else:
                raise ValueError("Combinação de ensaio A/B inválida ou incompleta.")

            Rp = Xp = Rs = Xs = ReqTotal = XeqTotal = None

            if circuit_type == "T":
                if referred_to == "primário":
                    Rp = Req / a2
                    Xp = Xeq / a2
                    Rs = Req - Rp
                    Xs = Xeq - Xp
                elif referred_to == "secundário":
                    Rp = Req * a2
                    Xp = Xeq * a2
                    Rs = Req - Rp
                    Xs = Xeq - Xp

            elif circuit_type == "L":
                if referred_to == "primário":
                    Rp = Req / a2
                    Xp = Xeq / a2
                    ReqTotal = Rp + (Req - Rp)
                    XeqTotal = Xp + (Xeq - Xp)
                elif referred_to == "secundário":
                    Rp = Req * a2
                    Xp = Xeq * a2
                    ReqTotal = Rp + (Req - Rp)
                    XeqTotal = Xp + (Xeq - Xp)

            elif circuit_type == "Série":
                ReqTotal = Req + (Rc_prime / a2 if Rc_prime else 0)
                XeqTotal = Xeq + (Xm_prime / a2 if Xm_prime else 0)

            st.markdown("### Resultados do Transformador")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Rc (Ω):** {round(Rc, 2)}")
                st.markdown(f"**Xm (Ω):** {round(Xm, 2)}")
                st.markdown(f"**Rc Referido (Ω):** {round(Rc_prime, 2) if Rc_prime else '—'}")
                st.markdown(f"**Xm Referido (Ω):** {round(Xm_prime, 2) if Xm_prime else '—'}")
                st.markdown(f"**Zphi (Ω):** {round(Zphi, 2)}")
                st.markdown(f"**Ic (mA):** {round(Ic * 1000, 2) if Ic else '—'}")
                st.markdown(f"**Im (mA):** {round(Im * 1000, 2) if Im else '—'}")

            with col2:
                st.markdown(f"**Req (Ω):** {round(Req, 2)}")
                st.markdown(f"**Xeq (Ω):** {round(Xeq, 2)}")
                st.markdown(f"**Rp (Ω):** {round(Rp, 2) if Rp else '—'}")
                st.markdown(f"**Xp (Ω):** {round(Xp, 2) if Xp else '—'}")
                st.markdown(f"**Rs (Ω):** {round(Rs, 2) if Rs else '—'}")
                st.markdown(f"**Xs (Ω):** {round(Xs, 2) if Xs else '—'}")
                st.markdown(f"**Req Total (Ω):** {round(ReqTotal, 2) if ReqTotal else '—'}")
                st.markdown(f"**Xeq Total (Ω):** {round(XeqTotal, 2) if XeqTotal else '—'}")

            st.markdown("### Diagrama Fasorial da Corrente de Excitação")

            if Ic is not None and Im is not None:
                fig, ax = plt.subplots()

                # Normaliza os vetores
                Iphi = np.sqrt(Ic**2 + Im**2)
                max_val = max(abs(Iphi), abs(Ic), abs(Im))
                fator_ampliacao = 1.2  # 20% extra para margem
                escala = 1 / max_val if max_val != 0 else 1

                # Vetores normalizados
                Ic_n = Ic * escala
                Im_n = Im * escala
                Iphi_n_x = Ic_n
                Iphi_n_y = Im_n

                # Vetor tensão (referência horizontal)
                ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='orange', label='V (referência)')

                # Correntes
                ax.quiver(0, 0, Ic_n, 0, angles='xy', scale_units='xy', scale=1, color='r', label='Ic (ativa)')
                ax.quiver(0, 0, 0, Im_n, angles='xy', scale_units='xy', scale=1, color='b', label='Im (reativa)')
                ax.quiver(0, 0, Iphi_n_x, Iphi_n_y, angles='xy', scale_units='xy', scale=1, color='g', label='Iφ (resultante)')

                # Adiciona o ângulo φ entre tensão e corrente
                if Iphi != 0:
                    cos_phi = Ic / Iphi
                    cos_phi = np.clip(cos_phi, -1, 1)
                    phi_rad = np.arccos(cos_phi)
                    phi_deg = np.degrees(phi_rad)

                    arc = Arc((0, 0), 0.4, 0.4, angle=0, theta1=0, theta2=phi_deg, color='gray', linestyle='--')
                    ax.add_patch(arc)
                    ax.text(0.3, 0.05, f'φ = {phi_deg:.2f}°', fontsize=10, color='gray')

                # Configurações do gráfico
                ax.set_xlim(-fator_ampliacao, fator_ampliacao)
                ax.set_ylim(-fator_ampliacao, fator_ampliacao)
                ax.set_xlabel("Eixo Real")
                ax.set_ylabel("Eixo Imaginário")
                ax.set_title("Diagrama Fasorial da Corrente de Excitação")
                ax.grid(True)
                ax.set_aspect('equal')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                st.pyplot(fig)
            else:
                st.warning("Corrente de excitação inválida ou ausente. Verifique os dados de entrada.")

        except ValueError as ve:
            st.error(f"Erro de entrada: {ve}")

        except ZeroDivisionError:
            st.error("Erro: Divisão por zero detectada.")

        except Exception as e:
            st.error(f"Erro inesperado: {e}")

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