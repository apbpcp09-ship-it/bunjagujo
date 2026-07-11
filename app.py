import streamlit as st
import torch
import torch.nn as nn
import random

st.set_page_config(page_title="분자 구조 AI 시뮬레이터", layout="centered")
st.title("🧪 분자 구조 예측 AI 시뮬레이터")
st.write("화학적 규칙(Domain Knowledge)이 주입된 LSTM 분자 생성 시뮬레이터입니다.")

VOCAB = ['<pad>', 'C', 'O', 'N', '=', '<eos>']
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}

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

if st.button("🚀 분자 생성 시뮬레이션 시작", use_container_width=True):
    with torch.no_grad():
        current_char = start_element
        generated_molecule = [current_char]
        current_input = torch.tensor([[char_to_idx[current_char]]])
        hidden = None
        
        # 화학적 꼼수 규칙 정의 (가짜 가중치 규칙)
        # 직전 원소에 따라 다음에 올 수 있는 '그럴듯한' 후보를 제한합니다.
        rules = {
            'C': ['=', 'O', 'N', 'C'],
            'O': ['C', '=', '<eos>'],
            'N': ['C', 'O', '<eos>'],
            '=': ['C', 'O', 'N']  # 이중결합 뒤에는 결합선이 또 못 오게 막음!
        }
        
        for step in range(5):
            output, hidden = model(current_input, hidden)
            
            # 1. AI가 예측한 값들의 순위를 가져옴
            logits = output.squeeze(0).squeeze(0)
            sorted_indices = torch.argsort(logits, descending=True)
            
            # 2. 화학 규칙 적용: 현재 원소 뒤에 올 수 있는 후보 필터링
            allowed_next = rules.get(current_char, VOCAB)
            
            # AI의 예측 중 규칙에 맞는 가장 높은 순위의 원소를 선택
            predicted_char = None
            predicted_idx = None
            for idx in sorted_indices:
                char = idx_to_char[idx.item()]
                if char in allowed_next and char != '<pad>':
                    predicted_char = char
                    predicted_idx = idx.item()
                    break
            
            # 만약 마땅한 게 없으면 무작위 규칙 후보 중 하나 선택
            if not predicted_char:
                predicted_char = random.choice(allowed_next)
                predicted_idx = char_to_idx[predicted_char]
                
            if predicted_char == '<eos>':
                break
                
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            current_char = predicted_char # 현재 상태 업데이트
            
        st.success("✨ 분자 구조 생성 완료!")
        result_str = " ➔ ".join([f"**[{char}]**" for char in generated_molecule])
        st.markdown(f"### 🧬 예측된 구조: {result_str}")
        st.info(f"💡 **시뮬레이션 분석**: 백지 상태의 LSTM 모델 예측에 '이중결합 뒤에 이중결합 금지', '산소 뒤의 결합 제한' 등의 화학 규칙 필터를 결합하여 현실적인 구조를 유도했습니다.")

