import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="다중 데이터 AI 시뮬레이터", layout="centered")
st.title("🧠 다중 데이터 실시간 AI 트레이닝 시뮬레이터")
st.write("여러 분자 구조를 동시에 입력받아 AI가 파국적 망각 없이 다양한 패턴을 골고루 학습합니다.")

VOCAB = ['<pad>', 'C', 'O', 'N', '=', '<eos>']
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

MAX_VALENCE = {'C': 4, 'O': 2, 'N': 3}

class MolecularLSTM(nn.Module):
    def __init__(self, vocab_size=6, embedding_dim=8, hidden_dim=16):
        super(MolecularLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

if 'model' not in st.session_state:
    st.session_state.model = MolecularLSTM()
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.05) # 안정적인 학습을 위해 lr 소폭 하향
    st.session_state.loss_history = []

model = st.session_state.model
optimizer = st.session_state.optimizer

st.sidebar.header("🕹️ 모드 선택 및 설정")
mode = st.sidebar.radio("원하는 작업을 선택하세요:", ["🏋️ 다중 분자 훈련시키기", "🚀 훈련된 AI로 분자 생성"])
temperature = st.sidebar.slider("🔥 AI 창의성 (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

def draw_molecule_with_hydrogen(molecule_list):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF', 'H': '#00AA00'}
    
    used_bonds = [0] * len(molecule_list)
    for idx, elem in enumerate(molecule_list):
        if elem == '=':
            if idx > 0: used_bonds[idx-1] += 2
            if idx < len(molecule_list) - 1: used_bonds[idx+1] += 2
        elif elem in ['C', 'O', 'N']:
            if idx > 0 and molecule_list[idx-1] != '=':
                used_bonds[idx] += 1
                used_bonds[idx-1] += 1

    x, y = 0.5, 0.0
    atom_positions = {}
    atom_idx = 0

    for idx, elem in enumerate(molecule_list):
        if elem == '=':
            ax.plot([x - 0.3, x + 0.3], [y + 0.06, y + 0.06], color='#444444', linewidth=3)
            ax.plot([x - 0.3, x + 0.3], [y - 0.06, y - 0.06], color='#444444', linewidth=3)
            x += 0.5
        else:
            if idx > 0 and molecule_list[idx-1] != '=':
                ax.plot([x - 0.7, x - 0.3], [y, y], color='#888888', linewidth=2)
            ax.text(x, y, elem, fontsize=22, fontweight='bold', color=colors.get(elem, '#000000'), ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
            atom_positions[atom_idx] = (x, y, elem, used_bonds[idx])
            atom_idx += 1
            x += 1.2

    for idx, (ax_x, ax_y, elem, u_bond) in atom_positions.items():
        needed_h = MAX_VALENCE[elem] - u_bond
        if needed_h <= 0: continue
        h_directions = []
        if idx == 0: h_directions.append((-0.4, 0))
        if idx == len(atom_positions) - 1: h_directions.append((0.4, 0))
        h_directions.extend([(0, 0.45), (0, -0.45)])
        for h_idx in range(min(needed_h, len(h_directions))):
            dx, dy = h_directions[h_idx]
            ax.plot([ax_x, ax_x + dx*0.6], [ax_y, ax_y + dy*0.6], color='#888888', linewidth=1.5)
            ax.text(ax_x + dx, ax_y + dy, 'H', fontsize=14, fontweight='bold', color=colors['H'], ha='center', va='center')

    ax.set_xlim(-0.2, x - 0.6)
    ax.set_ylim(-0.8, 0.8)
    ax.axis('off')
    st.pyplot(fig)

#
