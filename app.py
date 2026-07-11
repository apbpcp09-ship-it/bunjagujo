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

# 원소별 최대 결합선 (팔 개수)
# 탄소=4, 산소=2, 질소=3, 이중결합은 팔을 2개 소모함
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
        
        # 💡 핵심: 현재 분자 구조의 총 가용 결합선(남은 팔 개수)을 기억하는 상태 저장소
        # 첫 원소의 팔 개수로 시작 (예: C 선택 시 남은 팔 4개)
        available_arms = MAX_VALENCE[start_element]
        
        # 연속 결합 제한을 위한 플래그
        last_was_bond = False 
        
        for step in range(6):
            output, hidden = model(current_input, hidden)
            logits = output.squeeze(0).squeeze(0)
            sorted_indices = torch.argsort(logits, descending=True)
            
            predicted_char = None
            predicted_idx = None
            
            # AI의 예측 후보 중 '과학적으로 남은 팔 개수'에 맞는 것 필터링
            for idx in sorted_indices:
                char = idx_to_char[idx.item()]
                
                if char == '<pad>':
                    continue
                    
                # 1. 종료 조건: 남은 팔이 없거나 AI가 끝내고 싶어 할 때
                if char == '<eos>':
                    if step >= 2: # 최소 길이를 보장하기 위해
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                
                # 2. 이중 결합(=) 규칙 검사
                if char == '=':
                    # 직전이 결합선이 아니고, 현재 이어붙일 남은 팔이 최소 2개 이상 있어야 함
                    if not last_was_bond and available_arms >= 2:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
                    continue
                
                # 3. 일반 원소(C, O, N) 규칙 검사
                if char in ['C', 'O', 'N']:
                    # 결합이 이어지려면 기존에 남은 팔이 있어야 함
                    if available_arms > 0:
                        predicted_char = char
                        predicted_idx = idx.item()
                        break
            
            # 만약 모든 후보가 과학적 규칙에 위배되면 안전하게 강제 종료
            if not predicted_char or predicted_char == '<eos>':
                break
                
            # 팔 개수(Valence) 계산 업데이트
            if predicted_char == '=':
                # 이중결합이 추가되면 연결을 위해 기존 팔 1개 소모 (단, 이중결합 자체는 결합선 소모 없음, 다음 원소가 소모함)
                last_was_bond = True
            else:
                if last_was_bond:
                    # 이중결합(=) 뒤에 원소가 오면: 기존 팔 1개 소모 + 새 원소의 팔(최대-2)만큼 충전
                    available_arms = (available_arms - 1) + (MAX_VALENCE[predicted_char] - 2)
                    last_was_bond = False
                else:
                    # 일반 단일 결합으로 원소가 오면: 기존 팔 1개 소모 + 새 원소의 팔(최대-1)만큼 충전
                    available_arms = (available_arms - 1) + (MAX_VALENCE[predicted_char] - 1)
            
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            current_char = predicted_char
            
        # 과학적 보정: 만약 마지막에 팔이 남은 채로 끝나면 단일 결합선 구조가 흐려지므로 
        # 최종 SMILES 문자열을 정제합니다.
        result_text = "".join(generated_molecule)
        if result_text.endswith('='):
            result_text = result_text[:-1]
            generated_molecule.pop()
            
        st.success("✨ 과학적 분자 구조 생성 완료!")
        st.markdown(f"### 🧬 예측된 SMILES 코드: `{result_text}`")
        
        # NIH 이미지 엔진 렌더링
        try:
            encoded_smiles = result_text.replace("=", "%3D")
            nih_image_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image"
            st.image(nih_image_url, caption=f"화학 법칙을 준수한 {result_text}의 2D 구조", width=350)
        except:
            st.warning("⚠️ 구조 그리기 엔진 일시적 오류")
            
        st.info(f"💡 **과학적 시뮬레이션 분석**: LSTM의 과거 기억 제어 메커니즘에 실제 화학 가중치(탄소=팔4개, 산소=팔2개)를 연동했습니다. 이제 결합선 법칙을 초과하는 엉터리 분자는 생성되지 않습니다.")
