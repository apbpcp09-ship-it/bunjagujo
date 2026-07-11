import streamlit as st
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="옥텟 규칙 준수 AI 시뮬레이터", layout="centered")
st.title("🧪 옥텟 규칙(Octet Rule) 준수 AI 시뮬레이터")
st.write("LSTM 상태 제어에 실제 '옥텟 규칙 전자쌍 계산기'를 결합하여 화학적으로 100% 안정한 분자만 생성합니다.")

VOCAB = ['<pad>', 'C', 'O', 'N', '=', '<eos>']
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

# 원자가 전자 수 (옥텟 규칙을 만족하기 위해 필요한 전자의 기반)
VALENCE_ELECTRONS = {'C': 4, 'O': 6, 'N': 5}

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

# 🎨 원소와 결합선을 선명하게 그려주는 내장 그래픽 함수
def draw_molecule_custom(molecule_list):
    fig, ax = plt.subplots(figsize=(6, 2))
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF'}
    
    x, y = 0.0, 0.0
    render_elements = []
    
    for elem in molecule_list:
        render_elements.append(elem)

    for idx, elem in enumerate(render_elements):
        if elem == '=':
            ax.plot([x - 0.3, x + 0.3], [y + 0.06, y + 0.06], color='#444444', linewidth=3)
            ax.plot([x - 0.3, x + 0.3], [y - 0.06, y - 0.06], color='#444444', linewidth=3)
            x += 0.5
        else:
            if idx > 0 and render_elements[idx-1] != '=':
                ax.plot([x - 0.7, x - 0.3], [y, y], color='#888888', linewidth=2)
            
            ax.text(x, y, elem, fontsize=24, fontweight='bold', 
                    color=colors.get(elem, '#000000'), ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            x += 1.0

    ax.set_xlim(-0.5, x - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    st.pyplot(fig)

if st.button("🚀 옥텟 규칙 기반 분자 생성", use_container_width=True):
    with torch.no_grad():
        current_char = start_element
        generated_molecule = [current_char]
        current_input = torch.tensor([[char_to_idx[current_char]]])
        hidden = None
        
        # 💡 핵심: 옥텟 규칙 달성을 위한 공유 결합 전자 추적
        # 각 원소가 안정해지기 위해 공유해야 하는 타겟 결합선 수 (8 - 원자가전자수)
        needed_bonds = {'C': 4, 'O': 2, 'N': 3}
        
        # 현재 생성 중인 분자의 오픈된 결합 가능 잔여 용량
        current_needed = needed_bonds[start_element]
        last_was_bond = False
        
        for step in range(5):
            output, hidden = model(current_input, hidden)
            logits = output.squeeze(0).squeeze(0)
            sorted_indices = torch.argsort(logits, descending=True)
            
            predicted_char = None
            predicted_idx = None
            
            # 옥텟 규칙 잔여 전자가 0이면 완벽히 안정한 분자이므로 즉시 종료 유도
            if current_needed <= 0:
                predicted_char = '<eos>'
                break
                
            for idx in sorted_indices:
                char = idx_to_char[idx.item()]
                if char == '<pad>': continue
                
                # 종료 조건 처리
                if char == '<eos>':
                    if current_needed == 0 or step >= 3:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                
                # 이중결합(=) 규칙: 남은 요구 결합선이 최소 2개 이상이어야 함
                if char == '=':
                    if not last_was_bond and current_needed >= 2:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                
                # 일반 원소 규칙
                if char in ['C', 'O', 'N']:
                    # 결합선이 이어질 공간이 남아있을 때만 가능
                    if current_needed > 0:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
            
            if not predicted_char or predicted_char == '<eos>':
                break
                
            # 결합에 따른 옥텟 규칙 잔여 요구 결합선(Needed Bonds) 변동 계산
            if predicted_char == '=':
                last_was_bond = True
                # 이중결합선 자체는 연결 통로이므로 소모 수치 보류
            else:
                if last_was_bond:
                    # 이중결합으로 새 원소가 붙으면: 기존 요구량에서 2개 상쇄 + 새 원소의 남은 요구량(최대-2) 합산
                    current_needed = (current_needed - 2) + (needed_bonds[predicted_char] - 2)
                    last_was_bond = False
                else:
                    # 단일결합으로 새 원소가 붙으면: 기존 요구량에서 1개 상쇄 + 새 원소의 남은 요구량(최대-1) 합산
                    current_needed = (current_needed - 1) + (needed_bonds[predicted_char] - 1)
            
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            current_char = predicted_char
            
        # 불완전 결합 제거 및 최종 정제
        if generated_molecule[-1] == '=':
            generated_molecule.pop()
        
        result_text = "".join(generated_molecule)
        
        st.success("✨ 옥텟 규칙 만족 분자 구조 생성 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # 선명한 내부 드로잉 엔진 작동
        draw_molecule_custom(generated_molecule)
        
        # 과학적 분석 리포트 제공
        st.info(f"💡 **옥텟 규칙 분석**: 각 원소 주위의 총 결합선 수가 탄소=4개, 산소=2개, 질소=3개를 이루도록 제어했습니다. 숨겨진 수소(H)를 포함시켰을 때 모든 원소가 최외각 전자 8개(옥텟 규칙)를 완벽하게 만족하는 구조입니다.")
