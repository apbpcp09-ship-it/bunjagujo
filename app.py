import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="실시간 AI 트레이닝 시뮬레이터", layout="centered")
st.title("🧠 실시간 AI 분자 학습 & 트레이닝 시뮬레이터")
st.write("사용자가 직접 데이터를 먹여 AI의 가중치(Weights)를 실시간으로 훈련시키는 진짜 딥러닝 시스템입니다.")

VOCAB = ['<pad>', 'C', 'O', 'N', '=', '<eos>']
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

MAX_VALENCE = {'C': 4, 'O': 2, 'N': 3}

# 1. AI 모델 정의
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

# 💡 세션 스테이트를 이용해 새로고침되어도 AI의 가중치(기억)가 유지되도록 고정
if 'model' not in st.session_state:
    st.session_state.model = MolecularLSTM()
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.1) # 높은 학습률로 변화가 눈에 보이게 설정
    st.session_state.loss_history = []

model = st.session_state.model
optimizer = st.session_state.optimizer

# 2. 사이드바 - 두 개의 모드 제공 (훈련하기 vs 테스트하기)
st.sidebar.header("🕹️ 모드 선택 및 설정")
mode = st.sidebar.radio("원하는 작업을 선택하세요:", ["🏋️ 실시간 AI 훈련시키기", "🚀 훈련된 AI로 분자 생성"])
temperature = st.sidebar.slider("🔥 AI 창의성 (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

# 🎨 수소 시각화 함수
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

# --- 1모드: 실시간 AI 훈련시키기 ---
if mode == "🏋️ 실시간 AI 훈련시키기":
    st.subheader("🏋️ AI에게 정답 분자 학습시키기")
    st.write("AI에게 가르치고 싶은 결합 패턴을 입력하세요. AI가 오차를 계산해 즉시 뇌(가중치)를 업데이트합니다.")
    
    # 학습 데이터 입력창 예시 제시
    train_input = st.text_input("가르칠 정답 SMILES 입력 (예: C=C, COO, O=C=O, CNN):", "O=C=O")
    epochs = st.slider("반복 훈련 횟수 (Epochs)", min_value=5, max_value=50, value=20, step=5)
    
    if st.button("🔥 이 데이터로 실시간 딥러닝 시작"):
        # 입력 문자를 인덱스로 변환
        try:
            tokens = [c for c in train_input] + ['<eos>']
            indices = [char_to_idx[t] for t in tokens]
        except KeyError:
            st.error("⚠️ VOCAB에 없는 문자(C, O, N, =, <eos> 외)가 포함되어 있습니다.")
            st.stop()
            
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        
        # 실제 훈련 루프 (Backpropagation 작동)
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 입력 데이터 x와 정답 레이블 y 생성
            x_data = torch.tensor([indices[:-1]]).long() # 예: O, =, C, =, O
            y_data = torch.tensor([indices[1:]]).long()  # 정답: =, C, =, O, <eos>
            
            output, _ = model(x_data)
            
            # Loss 계산 및 역전파
            loss = loss_fn(output.view(-1, 6), y_data.view(-1))
            loss.backward()
            optimizer.step() # 가중치 실시간 업데이트!
            
            st.session_state.loss_history.append(loss.item())
            
        st.success(f"🎉 훈련 완료! AI가 '{train_input}' 구조를 학습하여 가중치를 조정했습니다.")
        
        # 학습 현황 그래프 시각화
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(st.session_state.loss_history[-epochs:], color='purple', label='Training Loss')
        ax.set_title("실시간 Loss(오차) 감소 추이")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)
        st.info("💡 오차(Loss)가 아래로 떨어질수록 AI가 방금 입력한 분자 구조의 패턴을 더 완벽하게 기억했다는 뜻입니다!")

# --- 2모드: 훈련된 AI로 분자 생성 ---
elif mode == "🚀 훈련된 AI로 분자 생성":
    st.subheader("🚀 조교된 AI의 실시간 추론 실력 확인")
    st.write("훈련을 많이 시킬수록, AI가 방금 가르쳐준 패턴과 유사한 분자를 우선적으로 뽑아내기 시작합니다.")
    
    start_element = st.selectbox("시작할 원소를 선택하세요:", ['C', 'O', 'N'])
    
    if st.button("🚀 훈련된 AI 기반 분자 생성", use_container_width=True):
        model.eval()
        with torch.no_grad():
            current_char = start_element
            generated_molecule = [current_char]
            current_input = torch.tensor([[char_to_idx[current_char]]])
            hidden = None
            
            needed_bonds = {'C': 4, 'O': 2, 'N': 3}
            current_needed = needed_bonds[start_element]
            last_was_bond = False
            
            for step in range(5):
                output, hidden = model(current_input, hidden)
                logits = output.squeeze(0).squeeze(0)
                
                # 가중치 확률 반영 (Temperature)
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_indices = torch.argsort(probs, descending=True)
                
                predicted_char = None
                predicted_idx = None
                
                if current_needed <= 0:
                    predicted_char = '<eos>'
                    break
                    
                for idx in sorted_indices:
                    char = idx_to_char[idx.item()]
                    if char == '<pad>': continue
                    if char == '<eos>':
                        if current_needed == 0 or step >= 2:
                            predicted_char = char
                            predicted_idx = idx.item()
                            break
                        continue
                    if char == '=':
                        if not last_was_bond and current_needed >= 2:
                            predicted_char = char
                            predicted_idx = idx.item()
                            break
                        continue
                    if char in ['C', 'O', 'N']:
                        if current_needed > 0:
                            predicted_char = char
                            predicted_idx = idx.item()
                            break
                
                if not predicted_char or predicted_char == '<eos>':
                    break
                    
                if predicted_char == '=':
                    last_was_bond = True
                else:
                    if last_was_bond:
                        current_needed = (current_needed - 2) + (needed_bonds[predicted_char] - 2)
                        last_was_bond = False
                    else:
                        current_needed = (current_needed - 1) + (needed_bonds[predicted_char] - 1)
                
                generated_molecule.append(predicted_char)
                current_input = torch.tensor([[predicted_idx]])
                current_char = predicted_char
                
            if generated_molecule[-1] == '=':
                generated_molecule.pop()
            
            result_text = "".join(generated_molecule)
            st.success("✨ 분자 생성 완료!")
            st.markdown(f"### 🧬 AI가 예측한 구조: `{result_text}`")
            
            draw_molecule_with_hydrogen(generated_molecule)
