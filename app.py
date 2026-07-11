import streamlit as st
import torch
import torch.nn as nn
import random

st.set_page_config(page_title="분자 구조 AI 시뮬레이터", layout="centered")
st.title("🔬 과학적 결합 규칙 기반 AI 시뮬레이터")
st.write("LSTM의 상태 제어 개념을 원자의 결합선(Valence) 추적 알고리즘과 결합하여 진짜 성립 가능한 분자를 만듭니다.")

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
            
        st.success("✨ 과학적 분자 구조 생성 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # 💡 탄소(C) 글자를 생략하지 않고 모두 보여주는 PubChem 이미지 API로 교체!
        try:
            encoded_smiles = result_text.replace("=", "%3D")
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/PNG"
            st.image(pubchem_url, caption=f"모든 원소 기호가 표시된 {result_text}의 구조", width=350)
        except:
            st.warning("⚠️ 구조 이미지 엔진 오류")
