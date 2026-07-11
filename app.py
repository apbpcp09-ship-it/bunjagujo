import streamlit as st
import torch
import torch.nn as nn
import random

st.set_page_config(page_title="분자 구조 AI 시뮬레이터", layout="centered")
st.title("🧪 분자 구조 예측 & 시각화 AI")
st.write("LSTM의 기억 창고로 구조를 예측하고, 분자 이미지를 실시간으로 시각화합니다.")

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

if st.button("🚀 분자 생성 및 시각화 시작", use_container_width=True):
    with torch.no_grad():
        current_char = start_element
        generated_molecule = [current_char]
        current_input = torch.tensor([[char_to_idx[current_char]]])
        hidden = None
        
        # 화학적 규칙 템플릿
        rules = {
            'C': ['=', 'O', 'N', 'C'],
            'O': ['C', '=', '<eos>'],
            'N': ['C', 'O', '<eos>'],
            '=': ['C', 'O', 'N']
        }
        
        for step in range(5):
            output, hidden = model(current_input, hidden)
            logits = output.squeeze(0).squeeze(0)
            sorted_indices = torch.argsort(logits, descending=True)
            
            allowed_next = rules.get(current_char, VOCAB)
            predicted_char = None
            predicted_idx = None
            
            for idx in sorted_indices:
                char = idx_to_char[idx.item()]
                if char in allowed_next and char != '<pad>':
                    predicted_char = char
                    predicted_idx = idx.item()
                    break
            
            if not predicted_char:
                predicted_char = random.choice(allowed_next)
                predicted_idx = char_to_idx[predicted_char]
                
            if predicted_char == '<eos>':
                break
                
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            current_char = predicted_char
            
        st.success("✨ 분자 구조 생성 완료!")
        
        # 1. 텍스트 결과 출력
        result_text = "".join(generated_molecule)
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # 2. NIH(미국 국립보건원) 공식 오픈 API를 사용하여 절대 깨지지 않는 2D 구조 그리기
        try:
            # '=' 기호가 웹 주소에서 깨지지 않도록 변환
            encoded_smiles = result_text.replace("=", "%3D")
            # NIH Cactus 서비스 URL 사용
            nih_image_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image"
            
            # 스트림릿 화면에 이미지 띄우기
            st.image(nih_image_url, caption=f"AI가 예측한 {result_text}의 2D 화학 구조", width=350)
            
        except Exception as e:
            st.warning("⚠️ 이미지를 불러오는 도중 오류가 발생했습니다. 잠시 후 다시 시도해 보세요.")
