import streamlit as st
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="옥텟 규칙 & 수소 시각화 AI", layout="centered")
st.title("🧪 옥텟 규칙 준수 & 수소 시각화 AI")
st.write("생략되던 수소(H) 원자까지 화학 결합 법칙에 맞게 계산하여 화면에 모두 표시합니다.")

VOCAB = ['<pad>', 'C', 'O', 'N', '=', '<eos>']
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

# 원소별 최대 결합선 (팔 개수)
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
tf_ratio = st.sidebar.slider("교사 강요 비율", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
start_element = st.selectbox("시작할 원소를 선택하세요:", ['C', 'O', 'N'])

# 🎨 수소(H)를 계산해서 사방으로 그려주는 고성능 시각화 함수
def draw_molecule_with_hydrogen(molecule_list):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF', 'H': '#00AA00'}
    
    # 1. 각 원소가 실제로 쓴 팔의 개수를 계산하기 위한 사전 준비
    # 실제 원소들만 추출 (이중결합 기호 제외)
    atoms_only = [elem for elem in molecule_list if elem != '=']
    
    # 각 원소 위치별 소모한 결합선 계산
    used_bonds = [0] * len(molecule_list)
    
    # 결합선 연결 정보 스캔
    for idx, elem in enumerate(molecule_list):
        if elem == '=':
            if idx > 0: used_bonds[idx-1] += 2
            if idx < len(molecule_list) - 1: used_bonds[idx+1] += 2
        elif elem in ['C', 'O', 'N']:
            # 앞 원소와의 단일 결합 체크
            if idx > 0 and molecule_list[idx-1] != '=':
                used_bonds[idx] += 1
                used_bonds[idx-1] += 1

    x, y = 0.5, 0.0
    atom_positions = {} # 원소들의 x 좌표 저장용
    atom_idx = 0

    # 메인 뼈대 그리기
    for idx, elem in enumerate(molecule_list):
        if elem == '=':
            ax.plot([x - 0.3, x + 0.3], [y + 0.06, y + 0.06], color='#444444', linewidth=3)
            ax.plot([x - 0.3, x + 0.3], [y - 0.06, y - 0.06], color='#444444', linewidth=3)
            x += 0.5
        else:
            if idx > 0 and molecule_list[idx-1] != '=':
                ax.plot([x - 0.7, x - 0.3], [y, y], color='#888888', linewidth=2)
            
            # 주 원소 텍스트 배치
            ax.text(x, y, elem, fontsize=22, fontweight='bold', 
                    color=colors.get(elem, '#000000'), ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
            
            atom_positions[atom_idx] = (x, y, elem, used_bonds[idx])
            atom_idx += 1
            x += 1.2

    # 2. 💡 수소(H) 붙이기 알고리즘 작동!
    for idx, (ax_x, ax_y, elem, u_bond) in atom_positions.items():
        # 필요한 총 팔 개수에서 이미 뼈대에 써버린 팔 개수를 빼면 필요한 수소 개수가 나옴
        needed_h = MAX_VALENCE[elem] - u_bond
        
        if needed_h <= 0:
            continue
            
        # 수소를 배치할 공간 설정 (위, 아래, 혹은 좌우 끝)
        h_directions = []
        if idx == 0:  # 맨 왼쪽 원소면 왼쪽 공간 활용
            h_directions.append((-0.4, 0))
        if idx == len(atom_positions) - 1: # 맨 오른쪽 원소면 오른쪽 공간 활용
            h_directions.append((0.4, 0))
        
        # 위, 아래 방향 추가
        h_directions.extend([(0, 0.45), (0, -0.45)])
        
        # 필요한 만큼 수소와 결합선 그리기
        for h_idx in range(min(needed_h, len(h_directions))):
            dx, dy = h_directions[h_idx]
            
            # 수소 결합선 그리기
            ax.plot([ax_x, ax_x + dx*0.6], [ax_y, ax_y + dy*0.6], color='#888888', linewidth=1.5)
            
            # 수소 'H' 글자 표시
            ax.text(ax_x + dx, ax_y + dy, 'H', fontsize=14, fontweight='bold',
                    color=colors['H'], ha='center', va='center')

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
            sorted_indices = torch.argsort(logits, descending=True)
            
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
        st.success("✨ 옥텟 규칙 만족 및 수소 결합 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # 🎨 수소 표시 드로잉 함수 실행!
        draw_molecule_with_hydrogen(generated_molecule)
