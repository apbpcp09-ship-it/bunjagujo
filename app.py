import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

st.set_page_config(page_title="창의성 0 모드 AI 시뮬레이터", layout="centered")
st.title("🧠 AI 날것의 뇌 & 창의성 0.0 시뮬레이터")
st.write("AI 창의성 수치를 0.0까지 낮출 수 있습니다. 확률 0%의 완벽한 결정론적 AI 모드를 체감해 보세요!")

# ----------------- 1. 사전 정의 및 AI 모델 설정 -----------------
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


# ----------------- 2. 사이드바 제어 및 설정 (0.0 설정 가능) -----------------
st.sidebar.header("🕹️ 모드 선택 및 설정")
mode = st.sidebar.radio("원하는 작업을 선택하세요:", ["🏋️ 다중 분자 훈련시키기", "🚀 훈련된 AI로 분자 생성"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ AI 하이퍼파라미터")

# 💡 최하 수치를 0.0으로 변경!
temperature = st.sidebar.slider("🔥 AI 창의성 (Temperature)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
use_filter = st.sidebar.checkbox("🛡️ 과학적 안전 필터(옥텟 규칙) 작동", value=True)


# ----------------- 3. 메인 화면 UI 요소 배치 -----------------
st.subheader("📝 AI 조작 패널")
train_input = st.text_input("정답 분자 데이터셋 입력 (쉼표로 구분):", "C=C, O=C=O, CNN, CON, C=O, N=N, C=N, N=C=N, C=C=O, O=N, O=C=N")
epochs = st.slider("반복 훈련 횟수 (Epochs)", min_value=10, max_value=100, value=50, step=10)
start_element = st.selectbox("생성 시작할 원소 선택:", ['C', 'O', 'N'])


# ----------------- 4. 시각화 엔진 -----------------
def draw_raw_molecule(molecule_list, has_filter):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF', 'H': '#00AA00'}
    
    used_bonds = [0] * len(molecule_list)
    if has_filter:
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
            
            ax.text(x, y, elem, fontsize=22, fontweight='bold', color=colors.get(elem, '#777777'), 
                    ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
            atom_positions[atom_idx] = (x, y, elem, used_bonds[idx])
            atom_idx += 1
            x += 1.2

    if has_filter:
        for idx, (ax_x, ax_y, elem, u_bond) in atom_positions.items():
            if elem not in MAX_VALENCE: continue
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


# ----------------- 5. 실행 로직 분기 -----------------
st.markdown("---")

if mode == "🏋️ 다중 분자 훈련시키기":
    st.info("현재 모드: AI에게 분자 데이터셋을 학습시킵니다. 아래 버튼을 누르세요.")
    if st.button("🔥 멀티 데이터셋 딥러닝 시작", use_container_width=True):
        molecule_examples = [s.strip() for s in train_input.split(",") if s.strip()]
        if not molecule_examples:
            st.error("⚠️ 올바른 분자식을 입력해 주세요.")
            st.stop()
            
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for example in molecule_examples:
                try:
                    tokens = [c for c in example] + ['<eos>']
                    indices = [char_to_idx[t] for t in tokens]
                except KeyError:
                    continue
                    
                optimizer.zero_grad()
                x_data = torch.tensor([indices[:-1]]).long()
                y_data = torch.tensor([indices[1:]]).long()
                
                output, _ = model(x_data)
                loss = loss_fn(output.view(-1, 6), y_data.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            st.session_state.loss_history.append(epoch_loss / len(molecule_examples))
            
        st.success(f"🎉 멀티 학습 완료! AI 뇌가 {len(molecule_examples)}개의 패턴을 골고루 흡수했습니다.")
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(st.session_state.loss_history[-epochs:], color='blue', label='Batch Loss')
        ax.set_title("멀티 데이터셋 훈련 오차 감소 현황")
        st.pyplot(fig)

elif mode == "🚀 훈련된 AI로 분자 생성":
    st.info("현재 모드: 학습된 가중치와 선택한 창의성 수치를 바탕으로 새로운 분자를 생성합니다.")
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
                
                # 💡 핵심: 창의성이 0.0일 때는 무조건 최고 높은 값(Argmax)만 추적하게 분기 처리!
                if temperature == 0.0:
                    # 주사위를 아예 던지지 않고 1순위 인덱스 딱 하나만 낚아챔
                    sampled_indices = torch.tensor([torch.argmax(logits).item()])
                else:
                    scaled_logits = logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    sampled_indices = torch.multinomial(probs, num_samples=10, replacement=True)
                
                predicted_char = None
                predicted_idx = None
                
                if current_needed <= 0 and use_filter:
                    predicted_char = '<eos>'
                    break
                
                if use_filter and temperature > 0.0:
                    for idx in sampled_indices:
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
                else:
                    # 필터가 꺼졌거나, 창의성이 0.0인 앵무새 모드일 때는 1순위 바로 선택
                    first_sample = sampled_indices[0].item()
                    predicted_char = idx_to_char[first_sample]
                    predicted_idx = first_sample

                if not predicted_char or predicted_char == '<eos>':
                    break
                    
                if predicted_char == '=':
                    last_was_bond = True
                else:
                    if last_was_bond:
                        current_needed = (current_needed - 2) + (needed_bonds.get(predicted_char, 0) - 2)
                        last_was_bond = False
                    else:
                        current_needed = (current_needed - 1) + (needed_bonds.get(predicted_char, 0) - 1)
                
                generated_molecule.append(predicted_char)
                current_input = torch.tensor([[predicted_idx]])
                current_char = predicted_char
                
            if generated_molecule[-1] == '=' and use_filter:
                generated_molecule.pop()
            
            result_text = "".join(generated_molecule)
            st.success("✨ 분자 생성 완료!")
            st.markdown(f"### 🧬 AI가 예측한 구조: `{result_text}`")
            
            draw_raw_molecule(generated_molecule, use_filter)
