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
    # ======= INICIALIZAÇÃO DO ESTADO DA SESSÃO =======
    # Garante que as variáveis de estado existam para evitar erros.
    if 'tipo_lamina_sugerido' not in st.session_state:
        st.session_state.tipo_lamina_sugerido = None
    if 'a_min_sugerido' not in st.session_state:
        st.session_state.a_min_sugerido = 0.0

    # ======= ENTRADAS =======
    st.header("Dados de entrada")

    # Define o tipo de lâmina com base no estado da sessão ou no padrão
    default_lamina_type = "Padronizada"
    if st.session_state.tipo_lamina_sugerido:
        default_lamina_type = st.session_state.tipo_lamina_sugerido

    # Obtém o índice do tipo de lâmina padrão para o selectbox
    lamina_options = ["Padronizada", "Comprida"]
    default_index = lamina_options.index(default_lamina_type)

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

    W2 = st.number_input("Potência (W2) em VA", value=300)
    f = st.number_input("Frequência em Hz", value=50)
    tipo_de_lamina = st.selectbox("Tipo de Lâmina do Núcleo", lamina_options, index=default_index)

    materiais_nucleo = {
        "Aço Silício GNO (Padrão)": 11300,
        "Aço Silício GO (Alto Rendimento)": 17000,
        "Aço Doce (Baixo Carbono)": 9000,
        "Ferrite Magnético": 4250,
        "Nanocristalino": 15000,
        "Liga Amorfa (Metglas)": 13500,
        "Ferro Puro (99,8%)": 20000
    }

    # Seletor de material do núcleo
    material_escolhido = st.selectbox(
        "Material do Núcleo",
        options=list(materiais_nucleo.keys())
    )

    # ======= CÁLCULOS =======
    W1 = 1.1 * W2
    V1_list = [int(v.strip()) for v in V1_input.split("/") if v.strip().isdigit()]
    V2_list = [int(v.strip()) for v in V2_input.split("/") if v.strip().isdigit()]
    I1 = [round(W1 / v, 2) for v in V1_list]
    d = 3 if W2 <= 500 else 2.5 if W2 <= 1000 else 2
    S1 = [round(i / d, 2) for i in I1]
    I2 = [round(W2 / v, 2) for v in V2_list]
    S2 = [round(i / d, 2) for i in I2]

    awg_table = [
        {"AWG": 25, "area_mm2": 0.162}, {"AWG": 24, "area_mm2": 0.205},
        {"AWG": 23, "area_mm2": 0.258}, {"AWG": 22, "area_mm2": 0.326},
        {"AWG": 21, "area_mm2": 0.410}, {"AWG": 20, "area_mm2": 0.518},
        {"AWG": 19, "area_mm2": 0.653}, {"AWG": 18, "area_mm2": 0.823},
        {"AWG": 17, "area_mm2": 1.04}, {"AWG": 16, "area_mm2": 1.31},
        {"AWG": 15, "area_mm2": 1.65}, {"AWG": 14, "area_mm2": 2.08},
        {"AWG": 13, "area_mm2": 2.62}, {"AWG": 12, "area_mm2": 3.31},
        {"AWG": 11, "area_mm2": 4.17}, {"AWG": 10, "area_mm2": 5.26},
        {"AWG": 9,  "area_mm2": 6.63}, {"AWG": 8,  "area_mm2": 8.37},
        {"AWG": 7,  "area_mm2": 10.55}, {"AWG": 6,  "area_mm2": 13.30},
        {"AWG": 5,  "area_mm2": 16.80}, {"AWG": 4,  "area_mm2": 21.15},
        {"AWG": 3,  "area_mm2": 26.67}, {"AWG": 2,  "area_mm2": 33.62},
        {"AWG": 1,  "area_mm2": 42.41}, {"AWG": 0,  "area_mm2": 53.49},
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
    Sm = round(coef * math.sqrt((fator_tipo * W2) / f), 1)
    Sg = round(Sm * 1.1, 1)

    a_calculado = math.ceil(math.sqrt(Sg))
    a = max(a_calculado, st.session_state.a_min_sugerido)
    b = round(Sg / a)

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
    
    def selecionar_lamina(a, tipo):
        laminas = laminas_padronizadas if tipo == "Padronizada" else laminas_compridas
        for lamina in laminas:
            if a <= lamina["a_cm"]:
                return lamina
        return laminas[-1]

    numero_lamina = selecionar_lamina(a, tipo_de_lamina)
    a = numero_lamina["a_cm"]
    SgEfetivo = a * b
    SmEfetivo = round(SgEfetivo / 1.1, 2)

    B_max = materiais_nucleo[material_escolhido]

    x = 1e8 / (4.44 * B_max * f)
    EspVolt = round(x / SmEfetivo, 4)

    N1 = math.ceil(EspVolt * V1_list[0])
    N2 = math.ceil(EspVolt * V2_list[0] * 1.1)

    if len(S1) >= 2 and len(S2) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N1 * fio_awg_s1_2['area_mm2'] + N2 * fio_awg_s2_1['area_mm2'] + N2 * fio_awg_s2_2['area_mm2']
    elif len(S1) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N1 * fio_awg_s1_2['area_mm2'] + N2 * fio_awg_s2_1['area_mm2']
    elif len(S2) >= 2:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N2 * fio_awg_s2_1['area_mm2'] + N2 * fio_awg_s2_2['area_mm2']
    else:
        Scu = N1 * fio_awg_s1_1['area_mm2'] + N2 * fio_awg_s2_1['area_mm2']

    Sj = numero_lamina["secao_mm2"]
    executavel = Sj / Scu
    Pfe = numero_lamina["peso_kgcm"] * b
    lm = (2 * a) + (2 * b) + (0.5 * a * math.pi)
    Pcu = (Scu / 100 * lm * 9) / 1000

    # ======= RESULTADOS =======
    
    st.header("Resultados do Dimensionamento")

    # --- LINHA 1: Espiras e Lâmina ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Número de Espiras")
        st.write(f"**Primário:** `{N1} espiras`")
        st.write(f"**Secundário:** `{N2} espiras`")

    with col2:
        st.subheader("Lâmina do Núcleo")
        st.write(f"**Tipo:** `Nº {numero_lamina['numero']} ({tipo_de_lamina})`")
        st.write(f"**Seção da Janela (Sj):** `{numero_lamina['secao_mm2']} mm²`")
        st.write(f"**Quantidade (empilhamento b):** `{b:.1f} cm`")

    st.divider()

    # --- LINHA 2: Bitola e Dimensões ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Bitola dos Cabos (AWG)")
        st.write(f"**Primário 1 ({V1_list[0]}V):** `{fio_awg_s1_1['AWG']} ({fio_awg_s1_1['area_mm2']} mm²)`")
        if len(S1) >= 2:
            st.write(f"**Primário 2 ({V1_list[1]}V):** `{fio_awg_s1_2['AWG']} ({fio_awg_s1_2['area_mm2']} mm²)`")
        st.write(f"**Secundário 1 ({V2_list[0]}V):** `{fio_awg_s2_1['AWG']} ({fio_awg_s2_1['area_mm2']} mm²)`")
        if len(S2) >= 2:
            st.write(f"**Secundário 2 ({V2_list[1]}V):** `{fio_awg_s2_2['AWG']} ({fio_awg_s2_2['area_mm2']} mm²)`")

    with col4:
        st.subheader("Dimensões do Núcleo")
        st.write(f"**Largura da coluna central (a):** `{a} cm`")
        st.write(f"**Seção Geométrica Efetiva (Sg):** `{SgEfetivo:.2f} cm²`")
        st.write(f"**Seção Magnética Efetiva (Sm):** `{SmEfetivo:.2f} cm²`")

    st.divider()

    # --- LINHA 3: Peso e Viabilidade ---
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Peso Estimado")
        st.write(f"**Núcleo de ferro (Pfe):** `{Pfe:.2f} kg`")
        st.write(f"**Enrolamento de cobre (Pcu):** `{Pcu:.2f} kg`")

    with col6:
        st.subheader("Análise de Viabilidade")
        st.write(f"**Área total do cobre (Scu):** `{Scu:.2f} mm²`")
        st.write(f"**Área da janela da lâmina (Sj):** `{Sj} mm²`")
        st.metric(label="Relação Sj / Scu", value=f"{executavel:.2f}", help="Um valor >= 3 é considerado ideal.")

    st.divider()

    # ======= LÓGICA DE EXECUTABILIDADE =======
    if executavel >= 3:
        st.success("✅ **Transformador Executável:** A relação entre a área da janela e a área do cobre (Sj/Scu) está dentro do critério aceitável.")
        st.session_state.a_min_sugerido = 0.0
        st.session_state.tipo_lamina_sugerido = None
    else:
        st.warning(f"❌ **TRANSFORMADOR NÃO EXECUTÁVEL (Sj/Scu = {executavel:.2f})**")
        st.info("A área da janela do núcleo (Sj) é muito pequena para a quantidade de cobre (Scu) necessária. Para resolver, você pode:")

        col_acao1, col_acao2 = st.columns(2)
        
        with col_acao1:
            st.write("**Opção 1: Aumentar o tamanho do núcleo**")
            st.write("Escolher uma lâmina maior aumentará a área da janela (Sj).")
            
            laminas_atuais = laminas_padronizadas if tipo_de_lamina == "Padronizada" else laminas_compridas
            indice_atual = next((i for i, item in enumerate(laminas_atuais) if item["numero"] == numero_lamina["numero"]), None)
            
            if indice_atual is not None and indice_atual + 1 < len(laminas_atuais):
                proxima_lamina = laminas_atuais[indice_atual + 1]
                if st.button(f"Usar próxima lâmina (Nº {proxima_lamina['numero']})"):
                    st.session_state.a_min_sugerido = proxima_lamina['a_cm']
                    st.rerun()
            else:
                st.error("Não há lâmina maior disponível neste tipo.")

        with col_acao2:
            if tipo_de_lamina == "Padronizada":
                st.write("**Opção 2: Usar lâmina 'Comprida'**")
                st.write("Lâminas do tipo 'Comprida' oferecem uma seção de janela maior para o mesmo tamanho.")
                if st.button("Alternar para lâmina 'Comprida'"):
                    st.session_state.tipo_lamina_sugerido = "Comprida"
                    st.session_state.a_min_sugerido = 0.0
                    st.rerun()

    # ======= VISUALIZAÇÃO 3D =======

    def criar_seções_do_transformador(x, y, z, dx, dy, dz):
        return np.array([
            [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
            [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
        ])
    
    def rotacionar_transformador(vertices, angle_rad):
        if np.isclose(angle_rad, np.pi/2):
            rotation_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        elif np.isclose(angle_rad, 0):
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = np.array([
                [np.cos(angle_rad), 1, 0], [0, -np.cos(angle_rad), 1], [np.sin(angle_rad), np.cos(angle_rad), 0]
            ])
        if vertices.ndim == 1:
            vertices = vertices.reshape(1,3)
        return np.dot(vertices, rotation_matrix.T)
        
    def plot_transformador(fig, vertices, color='gray', opacity=0.6):
        vx, vy, vz = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        faces_quad = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        i_tri, j_tri, k_tri = [], [], []
        for face in faces_quad:
            i_tri.extend([face[0], face[0]])
            j_tri.extend([face[1], face[2]])
            k_tri.extend([face[2], face[3]])

        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz, i=i_tri, j=j_tri, k=k_tri,
            opacity=opacity, color=color, flatshading=True, alphahull=0
        ))
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        edge_x, edge_y, edge_z = [], [], []
        for p1_idx, p2_idx in edges:
            edge_x.extend([vertices[p1_idx,0], vertices[p2_idx,0], None])
            edge_y.extend([vertices[p1_idx,1], vertices[p2_idx,1], None])
            edge_z.extend([vertices[p1_idx,2], vertices[p2_idx,2], None])
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(color='darkslategrey', width=2.5), showlegend=False
        ))

    def desenhar_uma_bobina(fig, num_espiras, radius, z_min, z_max, center_x, center_y, color, name, angle_rad_para_rotacao):
        if num_espiras <= 0 or radius <= 0:
            return
        n_pontos = int(num_espiras * 20) + 10
        z_vals = np.linspace(z_min, z_max, n_pontos)
        angle_param = np.linspace(0, num_espiras * 2 * np.pi, n_pontos)
        x_coil_orig = center_x + radius * np.cos(angle_param)
        y_coil_orig = center_y + radius * np.sin(angle_param)
        points = np.vstack((x_coil_orig, y_coil_orig, z_vals)).T
        vertices_rot = rotacionar_transformador(points, angle_rad_para_rotacao)
        fig.add_trace(go.Scatter3d(
            x=vertices_rot[:,0], y=vertices_rot[:,1], z=vertices_rot[:,2],
            mode='lines', line=dict(color=color, width=3.5), name=name
        ))

    def add_espiras(fig, angle_rad_para_rotacao, a_param_geral,
                V1_voltages, V2_voltages, N1, N2, z_min_coil, z_max_coil,
                center_x_p, center_y_p, perna_dx_p, perna_dy_p,
                center_x_s, center_y_s, perna_dx_s, perna_dy_s):

        def calcular_radii_adjusted(num_coils, perna_dx, perna_dy, a_ref_dim):
            if num_coils == 0: return []
            folga_carretel_visual = a_ref_dim * 0.05
            raio_primeira_espira = max(perna_dx / 2.0, perna_dy / 2.0) + folga_carretel_visual
            if num_coils == 1:
                return [raio_primeira_espira + a_ref_dim * 0.02]
            espessura_pacote_total_visual = a_ref_dim * 0.20
            if num_coils > 3:
                espessura_pacote_total_visual = min(a_ref_dim * (0.15 + num_coils * 0.03), a_ref_dim * 0.5)
            raio_ultima_espira = raio_primeira_espira + espessura_pacote_total_visual
            if raio_primeira_espira <= 0.01: raio_primeira_espira = 0.01
            if raio_ultima_espira <= raio_primeira_espira:
                raio_ultima_espira = raio_primeira_espira + a_ref_dim * 0.05
            return np.linspace(raio_primeira_espira, raio_ultima_espira, num_coils).tolist()

        # A função para desenhar as bobinas agora usa 'n_espiras' para definir as voltas.
        # Os parâmetros relacionados à tensão para a geometria foram removidos.
        def desenhar_bobinas_coords(voltages, n_espiras, radii, z_min, z_max, c_x, c_y):
            all_coils_points = []
            # n_espiras é o mesmo para todos os enrolamentos de um mesmo tipo (primário/secundário)
            if n_espiras <= 0: return []

            for i, v_val in enumerate(voltages):
                if i >= len(radii) or radii[i] <= 0: continue
                r = radii[i]

                # Define o número de pontos para criar uma hélice suave
                # Ex: 20 pontos por espira, com um mínimo de 50 pontos.
                n_pontos = max(int(n_espiras * 20), 50)
                z_vals = np.linspace(z_min, z_max, n_pontos)

                # O ângulo final é diretamente proporcional ao número de espiras (N * 2 * PI)
                angle_param = np.linspace(0, n_espiras * 2 * np.pi, n_pontos)

                x_coil_orig = c_x + r * np.cos(angle_param)
                y_coil_orig = c_y + r * np.sin(angle_param)
                all_coils_points.append({"points": np.vstack((x_coil_orig, y_coil_orig, z_vals)).T, "voltage": v_val})
            return all_coils_points

        V1, V2 = [float(v) for v in V1_voltages], [float(v) for v in V2_voltages]
        
        if V1 and N1 > 0:
            radii_p = calcular_radii_adjusted(len(V1), perna_dx_p, perna_dy_p, a_param_geral)
            # Passa N1 para a função de desenho
            for idx, coil_info in enumerate(desenhar_bobinas_coords(V1, N1, radii_p, z_min_coil, z_max_coil, center_x_p, center_y_p)):
                vertices_rot = rotacionar_transformador(coil_info["points"], angle_rad_para_rotacao)
                fig.add_trace(go.Scatter3d(x=vertices_rot[:,0], y=vertices_rot[:,1], z=vertices_rot[:,2],
                    mode='lines', line=dict(color='brown', width=3.5), name=f'Primário {idx+1} ({N1} espiras)'))
        if V2 and N2 > 0:
            radii_s = calcular_radii_adjusted(len(V2), perna_dx_s, perna_dy_s, a_param_geral)
            # Passa N2 para a função de desenho
            for idx, coil_info in enumerate(desenhar_bobinas_coords(V2, N2, radii_s, z_min_coil, z_max_coil, center_x_s, center_y_s)):
                vertices_rot = rotacionar_transformador(coil_info["points"], angle_rad_para_rotacao)
                fig.add_trace(go.Scatter3d(x=vertices_rot[:,0], y=vertices_rot[:,1], z=vertices_rot[:,2],
                    mode='lines', line=dict(color='gold', width=3.5), name=f'Secundário {idx+1} ({N2} espiras)'))



    # ALTERAÇÃO: A função agora recebe 'tipo_lamina'
    def gerar_visu_transformador(angle_rad, a, b, V1_tensões, V2_tensões, N1, N2, tipo_lamina):
        fig = go.Figure()

        if tipo_lamina == "Comprida":
            janela_x = a * 0.70
        else: # Padronizada
            janela_x = a * 0.35

        largura_perna_central_x = a * 0.8
        profundidade_perna_central_y = b * 0.7
        coil_height_z = 0.8 * a
        z_min_coil_ref = (1.25 * a) - (coil_height_z / 2)

        pc_dx, pc_dy, pc_dz = largura_perna_central_x, profundidade_perna_central_y, coil_height_z
        pc_x = (1.5 * a) - (pc_dx / 2)
        pc_y = (b / 2) - (pc_dy / 2)
        pc_z = z_min_coil_ref

        pl_dx, pl_dy, pl_dz = pc_dx / 2, pc_dy, pc_dz
        
        ple_x = pc_x - janela_x - pl_dx
        ple_y = pc_y
        ple_z = pc_z
        centro_x_ple_orig = ple_x + pl_dx / 2
        centro_y_ple_orig = ple_y + pl_dy / 2

        pld_x = pc_x + pc_dx + janela_x
        pld_y = pc_y
        pld_z = pc_z
        centro_x_pld_orig = pld_x + pl_dx / 2
        centro_y_pld_orig = pld_y + pl_dy / 2
        
        espessura_base_topo_z = a * 0.3
        be_dx = (pld_x + pl_dx) - ple_x
        be_dy = pc_dy
        be_dz = espessura_base_topo_z
        be_x = ple_x
        be_y = pc_y
        be_z = pc_z - espessura_base_topo_z

        bi_dx, bi_dy, bi_dz = be_dx, pc_dy, espessura_base_topo_z
        bi_x, bi_y, bi_z = ple_x, pc_y, pc_z + pc_dz

        parts_definitions = [
            {"pos": [pc_x, pc_y, pc_z], "dims": [pc_dx, pc_dy, pc_dz]},
            {"pos": [ple_x, ple_y, ple_z], "dims": [pl_dx, pl_dy, pl_dz]},
            {"pos": [pld_x, pld_y, pld_z], "dims": [pl_dx, pl_dy, pl_dz]},
            {"pos": [be_x, be_y, be_z], "dims": [be_dx, be_dy, be_dz]},
            {"pos": [bi_x, bi_y, bi_z], "dims": [bi_dx, bi_dy, bi_dz]}
        ]
        
        for p_def in parts_definitions:
            part_vertices = criar_seções_do_transformador(p_def["pos"][0], p_def["pos"][1], p_def["pos"][2], p_def["dims"][0], p_def["dims"][1], p_def["dims"][2])
            rotated_part_vertices = rotacionar_transformador(part_vertices, angle_rad)
            plot_transformador(fig, rotated_part_vertices, color='rgb(150, 150, 160)', opacity=0.6)

        z_inicio_bobinas, z_fim_bobinas = pc_z, pc_z + pc_dz
        
        # Passa N1 e N2 para a função add_espiras
        add_espiras(fig, angle_rad, a, V1_tensões, V2_tensões, N1, N2, z_inicio_bobinas, z_fim_bobinas,
                    centro_x_pld_orig, centro_y_pld_orig, pl_dx, pl_dy,
                    centro_x_ple_orig, centro_y_ple_orig, pl_dx, pl_dy)

        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data',
                xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
                bgcolor='rgb(25, 25, 35)',
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5))
            ),
            width=900, height=750,
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig


    st.header("Visualização 3D Interativa")
    angle = np.radians(90)
    V1_floats = [float(v) for v in V1_list]
    V2_floats = [float(v) for v in V2_list]
    
    fig = gerar_visu_transformador(angle, a, b, V1_floats, V2_floats, N1, N2, tipo_de_lamina)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Desafio 2")
    VM = st.number_input("Tensão Máxima", value=325)
    N = st.number_input("Número de Espiras", value=850)
    freq = st.number_input("Frequência (Hz)", value=50)

    # 1. Adiciona um slider para o usuário escolher a duração da simulação em milissegundos
    tempo_final_ms = st.slider(
        "Duração da Simulação (ms)", 
        min_value=20,          # Valor mínimo de 20ms (1 ciclo em 50Hz)
        max_value=1000,        # Valor máximo de 1000ms (1 segundo)
        value=340,             # Valor padrão de 340ms
        step=10                # Incremento de 10 em 10 ms
    )

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

            # 2. Usa o valor do slider (convertido para segundos) para criar o array de tempo
            t = np.arange(0, tempo_final_ms / 1000, 1/3000)

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

    prim_side = "alto" if N1 > N2 else "baixo"
    sec_side = "alto" if prim_side == "baixo" else "baixo"

    circuit_type = st.selectbox("Tipo de Circuito Equivalente", ["", "T", "L", "Série"])
    referred_to = st.selectbox("Referido a", ["", "primário", "secundário"])
    sec_type = st.selectbox(f"Tipo de Ensaio do Secundário", ["", "circuito-aberto", "curto-circuito"])
    prim_type = "curto-circuito" if sec_type == "circuito-aberto" else "circuito-aberto"

    st.markdown("## Dados do Ensaio de Curto-Circuito")
    Va = st.number_input("Tensão Va (V)", min_value=0.0, value=220.0)
    Ia = st.number_input("Corrente Ia (A)", min_value=0.0, value=1.0)
    Pa = st.number_input("Potência Pa (W)", min_value=0.0, value=100.0)

    st.markdown("## Conjunto de Dados Circuito Aberto")
    Vb = st.number_input("Tensão Vb (V)", min_value=0.0, value=50.0)
    Ib = st.number_input("Corrente Ib (A)", min_value=0.0, value=5.0)
    Pb = st.number_input("Potência Pb (W)", min_value=0.0, value=200.0)

    if st.button("Calcular Parâmetros"):
            # Validar entradas
            if Ia <= 0 or Ib <= 0:
                raise ValueError("Corrente deve ser maior que zero.")
            if Pa <= 0 or Pb <= 0:
                raise ValueError("Potência deve ser maior que zero.")
            if Va <= 0 or Vb <= 0:
                raise ValueError("Tensão deve ser maior que zero.")

            # Relação de transformação
            def calcular_relacao_transformacao(N1, N2):
                return N1 / N2

            # Converte impedância para o lado escolhido
            def referir_impedancia(valor, a, lado_ensaio, lado_referido):
                if lado_ensaio == lado_referido:
                    return valor
                elif lado_referido == "primário":
                    return valor * a ** 2
                elif lado_referido == "secundário":
                    return valor / a ** 2
                return valor

            # Ensaio de circuito aberto
            def calcular_ensaio_circuito_aberto(Vb, Ib, Pb, a, lado_ensaio, lado_referido):
                if Ib == 0 or Vb == 0:
                    return 0, 0, 0, 0, 0

                Rc = Vb ** 2 / Pb if Pb > 0 else float('inf')
                Ic = Pb / Vb
                Im = math.sqrt(Ib ** 2 - Ic ** 2) if Ib > Ic else 0
                Xm = Vb / Im if Im != 0 else float('inf')
                Zphi = Vb / Ib if Ib != 0 else float('inf')

                # Converte Rc e Xm para o lado referido
                Rc_ref = referir_impedancia(Rc, a, lado_ensaio, lado_referido)
                Xm_ref = referir_impedancia(Xm, a, lado_ensaio, lado_referido)

                return Rc_ref, Xm_ref, Zphi, Ic, Im

            # Ensaio de curto-circuito
            def calcular_ensaio_curto_circuito(Va, Ia, Pa, a, lado_ensaio, lado_referido):
                if Ia == 0 or Va == 0:
                    return 0, 0, 0

                Req = Pa / Ia ** 2 if Ia != 0 else float('inf')
                Zcc = Va / Ia
                Xeq = math.sqrt(Zcc ** 2 - Req ** 2) if Zcc > Req else 0

                # Converte Req e Xeq para o lado referido
                Req_ref = referir_impedancia(Req, a, lado_ensaio, lado_referido)
                Xeq_ref = referir_impedancia(Xeq, a, lado_ensaio, lado_referido)

                return Req_ref, Xeq_ref, Zcc

            # Parâmetros equivalentes para tipos T ou L
            def calcular_parametros_equivalentes(circuit_type, Req, Xeq):
                if circuit_type == 'Série':
                    return Req, Xeq, None, None, None, None
                elif circuit_type in ['T', 'L']:
                    Rp = Req / 2
                    Xp = Xeq / 2
                    Rs = Req / 2
                    Xs = Xeq / 2
                    return None, None, Rp, Xp, Rs, Xs
                else:
                    return None, None, None, None, None, None
            
            # Relação de transformação
            a = calcular_relacao_transformacao(N1, N2)

            # Determina o lado de cada ensaio
            ensaio_ca_lado = "secundário" if sec_type == "circuito-aberto" else "primário"
            ensaio_cc_lado = "secundário" if sec_type == "curto-circuito" else "primário"

            # Calcula os resultados referidos ao lado selecionado
            Rc_prime, Xm_prime, Zphi, Ic, Im = calcular_ensaio_circuito_aberto(
                Vb, Ib, Pb, a, ensaio_ca_lado, referred_to
            )

            ReqTotal, XeqTotal, Zcc = calcular_ensaio_curto_circuito(
                Va, Ia, Pa, a, ensaio_cc_lado, referred_to
            )

            # Calcula os parâmetros individuais se necessário
            ReqTotal_out, XeqTotal_out, Rp, Xp, Rs, Xs = calcular_parametros_equivalentes(circuit_type, ReqTotal, XeqTotal)

            # Exibição dos resultados
            st.markdown("### Resultados do Transformador")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Resultados do Ensaio de Circuito Aberto")
                ensaio_ca_lado = "secundário" if sec_type == "circuito-aberto" else "primário"
                st.text(f"Obtido com circuito aberto no lado {ensaio_ca_lado}")

                st.markdown(f"**Rc (Ω)** — resistência do núcleo (perdas no ferro): {round(Rc_prime, 2)}")
                st.markdown(f"**Xm (Ω)** — reatância magnetizante (campo magnético): {round(Xm_prime, 2)}")
                st.markdown(f"**Zphi (Ω)** — impedância do circuito aberto: {round(Zphi, 2)}")
                st.markdown(f"**Ic (mA)** — corrente ativa do núcleo: {round(Ic * 1000, 2)}")
                st.markdown(f"**Im (mA)** — corrente reativa do núcleo: {round(Im * 1000, 2)}")

            with col2:
                st.subheader("Resultados do Ensaio de Curto-Circuito")
                ensaio_cc_lado = "secundário" if sec_type == "curto-circuito" else "primário"
                st.text(f"Obtido com curto-circuito no lado {ensaio_cc_lado}")

                if circuit_type == 'Série':
                    st.markdown(f"**Req Total (Ω)** — resistência equivalente total: {round(ReqTotal, 2) if ReqTotal else '—'}")
                    st.markdown(f"**Xeq Total (Ω)** — reatância equivalente total: {round(XeqTotal, 2) if XeqTotal else '—'}")
                else:
                    st.markdown(f"**Rp (Ω)** — resistência do primário: {round(Rp, 2) if Rp else '—'}")
                    st.markdown(f"**Xp (Ω)** — reatância do primário: {round(Xp, 2) if Xp else '—'}")
                    st.markdown(f"**Rs (Ω)** — resistência do secundário: {round(Rs, 2) if Rs else '—'}")
                    st.markdown(f"**Xs (Ω)** — reatância do secundário: {round(Xs, 2) if Xs else '—'}")         

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

with tab4:
    st.title("Desafio 4 - Regulação de Tensão")

    # Entradas
    I2      = st.number_input("Corrente de carga I2 (A)", value=5.0)
    V2      = st.number_input("Tensão no secundário V2 (V)", value=220.0)
    R_eq    = st.number_input("Resistência equivalente R_eq (Ω)", value=0.5)
    X_eq    = st.number_input("Reatância equivalente X_eq (Ω)", value=1.2)
    cos_phi = st.number_input("Fator de potência cosφ", value=0.85)

    # Cálculo do seno
    if abs(cos_phi) <= 1:
        sin_phi = np.sqrt(1 - cos_phi**2)
        sin_phi = -sin_phi if cos_phi < 0 else sin_phi
    else:
        sin_phi = 0.0

    # Regulação
    regulacao = (I2 * (R_eq * cos_phi + X_eq * sin_phi)) / V2 * 100
    st.write(f"Regulação estimada: **{regulacao:.2f}%**")

    # Vetores
    V2_vec = np.array([V2, 0])
    VR     = np.array([R_eq * I2 * cos_phi, R_eq * I2 * sin_phi])
    VX     = np.array([-X_eq * I2 * sin_phi, X_eq * I2 * cos_phi])
    V1     = V2_vec + VR + VX

    # Função para desenhar linha + anotação deslocada
    def draw_line_and_label(fig, start, vec, label, color, xshift, yshift):
        end = start + vec
        # Desenha apenas a linha
        fig.add_trace(go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
        ))
        # Adiciona o rótulo via annotation
        fig.add_annotation(
            x=end[0], y=end[1],
            xref="x", yref="y",
            text=label,
            font=dict(color=color, size=14),
            showarrow=False,
            xshift=xshift,
            yshift=yshift,
            align="center"
        )
        return end

    # Build figure
    fig = go.Figure()
    origin = np.array([0, 0])

    # Desenha vetores encadeados
    p1 = draw_line_and_label(fig, origin, V2_vec,    "V₂",        "lime",  xshift=0,  yshift=-10)
    p2 = draw_line_and_label(fig, p1,     VR,        "I·R_eq",    "cyan",  xshift=20,   yshift=10)
    p3 = draw_line_and_label(fig, p2,     VX,        "j·I·X_eq",  "orange",xshift=30,   yshift=0)
    _  = draw_line_and_label(fig, origin, V1,        "V₁ (aprox)", "red",  xshift=0, yshift=20)

    # Layout escuro
    fig.update_layout(
        title="Diagrama Fasorial",
        xaxis_title="Eixo Real",
        yaxis_title="Eixo Imaginário",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        width=800, height=600,
        xaxis=dict(color='white', showgrid=True, scaleanchor="y"),
        yaxis=dict(color='white', showgrid=True),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    st.plotly_chart(fig)