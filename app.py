import streamlit as st
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="분자 구조 AI 시뮬레이터", layout="centered")
st.title("🔬 과학적 결합 규칙 기반 AI 시뮬레이터")
st.write("외부 엔진 없이 내부 그래픽 라이브러리로 모든 원소를 깨짐 없이 100% 시각화합니다.")

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
tf_ratio = st.sidebar.slider("교사 강요 비율", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
start_element = st.selectbox("시작할 원소를 선택하세요:", ['C', 'O', 'N'])

# 🎨 내장 그래픽 라이브러리로 분자를 직접 그리는 함수
def draw_molecule_custom(molecule_list):
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # 원소별 색상 정의 (화학 표준 색상 반영)
    colors = {'C': '#222222', 'O': '#FF4D4D', 'N': '#3333FF'}
    
    x = 0.0
    y = 0.0
    
    # 텍스트에서 결합선과 원소를 분리하여 순서대로 그리기
    render_elements = []
    i = 0
    while i < len(molecule_list):
        if molecule_list[i] == '=':
            render_elements.append('=')
        else:
            render_elements.append(molecule_list[i])
        i += 1

    # 그리기 알고리즘
    for idx, elem in enumerate(render_elements):
        if elem == '=':
            # 이중 결합 선 2개 그리기
            ax.plot([x - 0.3, x + 0.3], [y + 0.05, y + 0.05], color='#555555', linewidth=3)
            ax.plot([x - 0.3, x + 0.3], [y - 0.05, y - 0.05], color='#555555', linewidth=3)
            x += 0.5
        else:
            # 직전이 원소였고 이중결합이 아니었다면 단일 결합선 먼저 그리기
            if idx > 0 and render_elements[idx-1] != '=':
                ax.plot([x - 0.7, x - 0.3], [y, y], color='#888888', linewidth=2)
            
            # 원소 글자 박기
            ax.text(x, y, elem, fontsize=24, fontweight='bold', 
                    color=colors.get(elem, '#000000'), ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            x += 1.0

    ax.set_xlim(-0.5, x - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off') # 격자나 축 숨기기
    st.pyplot(fig)

if st.button("🚀 과학적 분자 생성 시작", use_container_width=True):
    with torch.no_grad():
        current_char = start_element
        generated_molecule = [current_char]
        current_input = torch.tensor([[char_to_idx[current_char]]])
        hidden = None
        
        available_arms = MAX_VALENCE[start_element]
        last_was_bond = False 
        
        for step in range(6):
            output, hidden = model(current_input, hidden)
            logits = output.squeeze(0).squeeze(0)
            sorted_indices = torch.argsort(logits, descending=True)
            
            predicted_char = None
            predicted_idx = None
            
            for idx in sorted_indices:
                char = idx_to_char[idx.item()]
                if char == '<pad>': continue
                if char == '<eos>':
                    if step >= 2:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                if char == '=':
                    if not last_was_bond and available_arms >= 2:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                if char in ['C', 'O', 'N']:
                    if available_arms > 0:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
            
            if not predicted_char or predicted_char == '<eos>':
                break
                
            if predicted_char == '=':
                last_was_bond = True
            else:
                if last_was_bond:
                    available_arms = (available_arms - 1) + (MAX_VALENCE[predicted_char] - 2)
                    last_was_bond = False
                else:
                    available_arms = (available_arms - 1) + (MAX_VALENCE[predicted_char] - 1)
            
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            current_char = predicted_char
            
        result_text = "".join(generated_molecule)
        if result_text.endswith('='):
            result_text = result_text[:-1]
            generated_molecule.pop()
            
        st.success("✨ 과학적 분자 구조 생성 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # 🎨 우리가 직접 만든 깨짐 없는 시각화 함수 실행!
        draw_molecule_custom(generated_molecule)
