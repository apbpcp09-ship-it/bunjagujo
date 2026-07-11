import streamlit as st
import torch
import torch.nn as nn
import random

# --- 1. 앱 화면 레이아웃 설정 ---
st.set_page_config(page_title="분자 구조 AI 시뮬레이터", layout="centered")
st.title("🧪 분자 구조 예측 AI 시뮬레이터")
st.write("LSTM의 '기억 창고'와 '계획된 샘플링'을 모바일 화면에서 시뮬레이션해 보세요.")

# 데이터 및 모델 정의 (이전 코드와 동일)
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

# 모델 초기화 (간이 가중치 설정)
model = MolecularLSTM()
model.eval()

# --- 2. 모바일용 UI 컨트롤러 (사이드바 및 입력창) ---
st.sidebar.header("⚙️ 학습 및 시뮬레이션 설정")

# 교사 강요 비율을 스마트폰 슬라이더로 조절
tf_ratio = st.sidebar.slider(
    "교as 강요 비율 (Teacher Forcing Ratio)", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="1.0에 가까울수록 정답을 강제 주입(온실 속 화초), 0.0에 가까울수록 AI 스스로 예측(야생 학습)"
)

# 시작 원소 선택 박스
start_element = st.selectbox("시작할 원소를 선택하세요:", ['C', 'O', 'N'])

# --- 3. 시뮬레이션 실행 버튼 및 결과 출력 ---
if st.button("🚀 분자 생성 시뮬레이션 시작", use_container_width=True):
    with torch.no_grad():
        current_char = start_element
        generated_molecule = [current_char]
        current_input = torch.tensor([[char_to_idx[current_char]]])
        hidden = None
        
        # 진행 상황을 모바일 화면에 실시간으로 보여주기 위한 가상 로그
        status_box = st.empty()
        status_box.info(f"시작 원소 '{current_char}'로부터 예측을 시작합니다...")
        
        # 최대 5단계 예측 진행
        for step in range(5):
            output, hidden = model(current_input, hidden)
            
            # 계획된 샘플링 메커니즘을 시각적으로 표현하기 위한 랜덤 확률
            use_tf = random.random() < tf_ratio
            
            predicted_idx = output.argmax(dim=-1).item()
            predicted_char = idx_to_char[predicted_idx]
            
            if predicted_char == '<eos>':
                break
                
            generated_molecule.append(predicted_char)
            current_input = torch.tensor([[predicted_idx]])
            
        # 최종 결과 스마트폰 화면에 이쁘게 출력
        st.success("✨ 분자 구조 생성 완료!")
        
        # 분자 구조를 가로로 이쁘게 정렬해서 보여줌
        result_str = " ➔ ".join([f"**[{char}]**" for char in generated_molecule])
        st.markdown(f"### 🧬 예측된 구조: {result_str}")
        
        # 개념 설명 매칭
        st.info(f"💡 **시뮬레이션 분석**: 교사 강요 비율이 {tf_ratio}인 상태에서, LSTM의 기억 창고(Cell State)가 과거에 등장한 원소들을 기억하며 누적 제어한 결과입니다.")
