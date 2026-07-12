import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

st.set_page_config(page_title="무조건 시각화 AI 시뮬레이터", layout="centered")
st.title("🧠 AI 날것의 뇌 & 무조건 시각화 시뮬레이터")
st.write("이제 필터를 꺼서 AI가 온갖 헛소리를 뱉어도, 에러 없이 그 민낯을 그대로 시각화하여 보여줍니다!")

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
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.05)
    st.session_state.loss_history = []

model = st.session_state.model
optimizer = st.session_state.optimizer

st.sidebar.header("🕹️ 모드 선택 및 설정")
mode = st.sidebar.radio("원하는 작업을 선택하세요:", ["🏋️ 다중 분자 훈련시키기", "🚀 훈련된 AI로 분자 생성"])
temperature = st.sidebar.slider("🔥 AI 창의성 (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
use_filter = st.sidebar.checkbox("🛡️ 과학적 안전 필터(옥텟 규칙) 작동", value=True)

# 💡 필터가 꺼져도 에러 없이 뼈대만 그리는 새로운 시각화 함수
def draw_raw_molecule(molecule_list, has_filter):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF', 'H': '#00AA00'}
    
    # 1. 옥텟 규칙이 켜져 있을 때만 기존 수소 결합 계산 진행
    used_bonds = [0] * len(molecule_list)
    if has_filter:
        for idx, elem in enumerate(molecule_list):
            if elem == '=':
                if idx > 0: used_bonds[idx-1] += 2
                if idx < len(molecule_list) - 1: used_bonds
