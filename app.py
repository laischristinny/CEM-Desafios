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
            [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z], # Vértices 0,1,2,3 (base)
            [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz] # Vértices 4,5,6,7 (topo)
        ])
    
    def rotacionar_transformador(vertices, angle_rad):
        if np.isclose(angle_rad, np.pi/2): # Rotação específica x->y, y->z, z->x (se essa for a intenção)
            rotation_matrix = np.array([
                [0, 1, 0], # y_velho para x_novo
                [0, 0, 1], # z_velho para y_novo
                [1, 0, 0]  # x_velho para z_novo
            ])
        elif np.isclose(angle_rad, 0):
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = np.array([ # Matriz original do usuário
                [np.cos(angle_rad), 1, 0],
                [0, -np.cos(angle_rad), 1],
                [np.sin(angle_rad), np.cos(angle_rad), 0]
            ])
        if vertices.ndim == 1:
            vertices = vertices.reshape(1,3)
        return np.dot(vertices, rotation_matrix.T)
        
    def plot_transformador(fig, vertices, color='gray', opacity=0.6):
        vx = vertices[:, 0]
        vy = vertices[:, 1]
        vz = vertices[:, 2]

        # Faces do paralelepípedo (cada face é uma lista de 4 índices de vértices)
        # Ordem: Base, Topo, Frente, Trás, Esquerda, Direita
        faces_quad = [
            [0, 1, 2, 3],  # Base
            [4, 5, 6, 7],  # Topo (corrigido para usar 4,5,6,7 no sentido horário ou anti-horário consistente) -> 7,6,5,4 ou 4,5,6,7
            [0, 1, 5, 4],  # Frente
            [3, 2, 6, 7],  # Trás
            [0, 3, 7, 4],  # Esquerda
            [1, 2, 6, 5]   # Direita
        ]
        
        i_tri, j_tri, k_tri = [], [], []
        for face in faces_quad:
            # Triângulo 1 da face: v0, v1, v2
            i_tri.append(face[0])
            j_tri.append(face[1])
            k_tri.append(face[2])
            # Triângulo 2 da face: v0, v2, v3
            i_tri.append(face[0])
            j_tri.append(face[2])
            k_tri.append(face[3])

        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=i_tri, j=j_tri, k=k_tri,
            opacity=opacity, color=color, flatshading=True,
            alphahull=0 # Garante que usemos os triângulos definidos
        ))

        # Arestas para melhor definição
        edges = [
            (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        edge_x, edge_y, edge_z = [], [], []
        for p1_idx, p2_idx in edges:
            edge_x.extend([vertices[p1_idx,0], vertices[p2_idx,0], None])
            edge_y.extend([vertices[p1_idx,1], vertices[p2_idx,1], None])
            edge_z.extend([vertices[p1_idx,2], vertices[p2_idx,2], None])
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(color='darkslategrey', width=2.5), showlegend=False
        ))


    def add_espiras(fig, angle_rad_para_rotacao, a_param_geral, # Passando 'a' para referência de tamanho
                V1_voltages, V2_voltages,
                z_min_coil, z_max_coil,
                center_x_p, center_y_p, perna_dx_p, perna_dy_p,
                center_x_s, center_y_s, perna_dx_s, perna_dy_s):

        def calcular_max_voltage(V1, V2):
            all_v = V1 + V2
            if not all_v: return 1.0
            positivos = [v for v in all_v if v > 0]
            return max(max(positivos) if positivos else 1.0, 1.0)

        def calcular_radii_adjusted(num_coils, perna_dx, perna_dy, a_ref_dim):
            if num_coils == 0: return []

            # Folga entre a superfície da perna e a primeira camada de espiras (carretel visual)
            folga_carretel_visual = a_ref_dim * 0.05 # Ex: 5% de 'a'

            # O raio da primeira espira (a mais interna, mas externa à perna)
            # Este é o raio até o centro do "fio" da primeira espira.
            raio_primeira_espira = max(perna_dx / 2.0, perna_dy / 2.0) + folga_carretel_visual

            if num_coils == 1:
                # Se só uma espira, adicionamos uma pequena espessura visual para o fio
                return [raio_primeira_espira + a_ref_dim * 0.02] 

            # Espessura radial total do pacote de bobinas (todas as camadas)
            espessura_pacote_total_visual = a_ref_dim * 0.20 # Ex: 20% de 'a'
            if num_coils > 3: # Se muitas espiras, pode precisar de um pacote mais grosso
                espessura_pacote_total_visual = a_ref_dim * (0.15 + num_coils * 0.03) # Aumenta com o número
                espessura_pacote_total_visual = min(espessura_pacote_total_visual, a_ref_dim * 0.5) # Limita

            raio_ultima_espira = raio_primeira_espira + espessura_pacote_total_visual
            
            # Garante que os raios sejam positivos e que o último seja maior que o primeiro
            if raio_primeira_espira <= 0.01: raio_primeira_espira = 0.01 
            if raio_ultima_espira <= raio_primeira_espira:
                raio_ultima_espira = raio_primeira_espira + a_ref_dim * 0.05 # Garante um pacote mínimo

            return np.linspace(raio_primeira_espira, raio_ultima_espira, num_coils).tolist()

        def desenhar_bobinas_coords(voltages, radii, z_min, z_max, max_v_ref, base_angle_mult, c_x, c_y):
            all_coils_points = []
            for i, v_val in enumerate(voltages):
                if i >= len(radii): continue
                r = radii[i]
                if r <= 0: continue
                
                rel_v = v_val / max_v_ref
                min_scale, max_scale = 0.5, 1.5 # densidade não usada diretamente no angulo aqui
                # densidade = max(min_scale, min(max_scale, min_scale + (max_scale - min_scale) * rel_v))
                n_pontos = max(int(80 * rel_v) + 25, 25) # Aumentado para mais suavidade
                z_vals = np.linspace(z_min, z_max, n_pontos)
                # ângulo para ter voltas visíveis, densidade afeta quantas voltas
                angle_param = np.linspace(0, base_angle_mult * 2 * np.pi * (min_scale + (max_scale - min_scale) * rel_v), n_pontos)
                
                x_coil_orig = c_x + r * np.cos(angle_param)
                y_coil_orig = c_y + r * np.sin(angle_param)
                
                coil_points = np.vstack((x_coil_orig, y_coil_orig, z_vals)).T
                all_coils_points.append({"points": coil_points, "voltage": v_val})
            return all_coils_points

        V1 = [float(v) for v in V1_voltages]
        V2 = [float(v) for v in V2_voltages]
        max_voltage_ref = calcular_max_voltage(V1, V2)
        num_V1 = len(V1)
        num_V2 = len(V2)

        if num_V1 > 0:
            radii_p = calcular_radii_adjusted(num_V1, perna_dx_p, perna_dy_p, a_param_geral)
            coils_data_p = desenhar_bobinas_coords(V1, radii_p, z_min_coil, z_max_coil, max_voltage_ref, 8, center_x_p, center_y_p)
            for idx, coil_info in enumerate(coils_data_p):
                vertices_espiras_p_orig = coil_info["points"]
                if vertices_espiras_p_orig.size == 0: continue
                vertices_espiras_p_rot = rotacionar_transformador(vertices_espiras_p_orig, angle_rad_para_rotacao)
                fig.add_trace(go.Scatter3d(
                    x=vertices_espiras_p_rot[:,0], y=vertices_espiras_p_rot[:,1], z=vertices_espiras_p_rot[:,2],
                    mode='lines', line=dict(color='brown', width=3.5), name=f'Primário {idx+1}'
                ))

        if num_V2 > 0:
            radii_s = calcular_radii_adjusted(num_V2, perna_dx_s, perna_dy_s, a_param_geral)
            coils_data_s = desenhar_bobinas_coords(V2, radii_s, z_min_coil, z_max_coil, max_voltage_ref, 8, center_x_s, center_y_s)
            for idx, coil_info in enumerate(coils_data_s):
                vertices_espiras_s_orig = coil_info["points"]
                if vertices_espiras_s_orig.size == 0: continue
                vertices_espiras_s_rot = rotacionar_transformador(vertices_espiras_s_orig, angle_rad_para_rotacao)
                fig.add_trace(go.Scatter3d(
                    x=vertices_espiras_s_rot[:,0], y=vertices_espiras_s_rot[:,1], z=vertices_espiras_s_rot[:,2],
                    mode='lines', line=dict(color='gold', width=3.5), name=f'Secundário {idx+1}'
                ))


    def gerar_visu_transformador(angle_rad, a, b, V1_tensões, V2_tensões):
        fig = go.Figure()

        largura_perna_central_x = a * 0.8
        profundidade_perna_central_y = b * 0.7 # Relacionado a 'b'
        coil_height_z = 0.8 * a
        z_min_coil_ref = (1.25 * a) - (coil_height_z / 2)

        pc_dx = largura_perna_central_x
        pc_dy = profundidade_perna_central_y
        pc_dz = coil_height_z
        pc_x = (1.5 * a) - (pc_dx / 2)
        pc_y = (b / 2) - (pc_dy / 2)
        pc_z = z_min_coil_ref

        pl_dx = pc_dx / 2 
        pl_dy = pc_dy 
        pl_dz = pc_dz 
        janela_x = a * 0.35

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

        bi_dx = be_dx
        bi_dy = pc_dy
        bi_dz = espessura_base_topo_z
        bi_x = ple_x
        bi_y = pc_y
        bi_z = pc_z + pc_dz

        parts_definitions = [
            {"pos": [pc_x, pc_y, pc_z], "dims": [pc_dx, pc_dy, pc_dz]},
            {"pos": [ple_x, ple_y, ple_z], "dims": [pl_dx, pl_dy, pl_dz]},
            {"pos": [pld_x, pld_y, pld_z], "dims": [pl_dx, pl_dy, pl_dz]},
            {"pos": [be_x, be_y, be_z], "dims": [be_dx, be_dy, be_dz]},
            {"pos": [bi_x, bi_y, bi_z], "dims": [bi_dx, bi_dy, bi_dz]}
        ]
        core_color = 'rgb(150, 150, 160)' # Cor do núcleo
        core_opacity = 0.6 # Opacidade do núcleo

        for p_def in parts_definitions:
            part_vertices = criar_seções_do_transformador(
                p_def["pos"][0], p_def["pos"][1], p_def["pos"][2],
                p_def["dims"][0], p_def["dims"][1], p_def["dims"][2]
            )
            rotated_part_vertices = rotacionar_transformador(part_vertices, angle_rad)
            plot_transformador(fig, rotated_part_vertices, color=core_color, opacity=core_opacity)

        z_inicio_bobinas = pc_z 
        z_fim_bobinas = pc_z + pc_dz

        # Chamada para add_espiras agora passa o parâmetro 'a'
        add_espiras(fig, angle_rad, a, # <--- Passando 'a' aqui
                    V1_tensões, V2_tensões,
                    z_inicio_bobinas, z_fim_bobinas,
                    centro_x_pld_orig, centro_y_pld_orig, pl_dx, pl_dy,
                    centro_x_ple_orig, centro_y_ple_orig, pl_dx, pl_dy)

        fig.update_layout( 
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
                xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
                bgcolor='rgb(25, 25, 35)',
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5))
            ),
            width=900, height=750,
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig


    st.title("Transformador Monofásico - Visualização 3D Interativa")

    if st.button("Gerar Transformador"):
        angle = np.radians(90)
        V1_floats = [float(v) for v in V1_list]
        V2_floats = [float(v) for v in V2_list]
        fig = gerar_visu_transformador(angle, a, b, V1_floats, V2_floats)
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