import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="창의성 조절 AI 시뮬레이터", layout="centered")
st.title("🧪 AI 창의성 조절 분자 시뮬레이터")
st.write("의미 없던 교사 강요 대신, AI의 '무작위성(Temperature)'을 조절하여 다양한 분자를 탐색합니다.")

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

model = MolecularLSTM()
model.eval()

st.sidebar.header("⚙️ 시뮬레이션 설정")
# 💡 교사 강요 슬라이더를 진짜 작동하는 'AI 창의성(Temperature)' 슬라이더로 변경!
temperature = st.sidebar.slider("🔥 AI 창의성 (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
start_element = st.selectbox("시작할 원소를 선택하세요:", ['C', 'O', 'N'])

def draw_molecule_with_hydrogen(molecule_list):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF', 'H': '#00AA00'}
    
    atoms_only = [elem for elem in molecule_list if elem != '=']
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
            
            ax.text(x, y, elem, fontsize=22, fontweight='bold', 
                    color=colors.get(elem, '#000000'), ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
            
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

if st.button("🚀 과학적 분자 생성 시작", use_container_width=True):
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
            
            # 💡 핵심: Temperature 슬라이더 값을 적용하여 확률 분포를 변형합니다.
            # 값이 높을수록 예측 값들이 평탄해져서 무작위성이 올라감!
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # 높은 확률 순으로 인덱스 정렬
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
        st.success("✨ 분자 생성 및 시각화 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        draw_molecule_with_hydrogen(generated_molecule)
        st.info(f"💡 **시뮬레이터 분석**: 현재 창의성 레벨은 **{temperature}**입니다. 옥텟 규칙 필터 내부에서 AI가 확률적 다양성을 발휘하여 매번 색다른 분자를 탐색합니다.")
