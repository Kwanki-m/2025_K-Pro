import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform



# Streamlit sidebar for user inputs
st.sidebar.title("모델 하이퍼파라미터 설정")
m_d = st.sidebar.slider("Max depth", 1, 20, value=3)
n_e = st.sidebar.slider("n_estimators", 50, 500, value=100)
l_r = st.sidebar.slider("Learning Rate", 0.01, 1.0, value=0.1)
sb = st.sidebar.slider("Subsample", 0.1, 1.0, value=0.8)

# Sidebar 계산 버튼 생성
calculation_button = st.sidebar.button("계산")

def main():
    # 데이터 로드
    df = pd.read_csv("SN_total.csv")
    df = df.set_index("시간")
    df.index = pd.DatetimeIndex(df.index)
    
    # 원수 탁도와 응집제 주입률 로그 변환 추가
    df["로그 원수 탁도"] = np.log10(df["원수 탁도"])
    df["로그 응집제 주입률"] = np.log10(df["3단계 1계열 응집제 주입률"])

    # 타겟 변수 선택
    st.title("Target 변수 선택")
    col = st.selectbox("Target 변수를 선택하세요", df.columns[1:])
    st.dataframe(df[col])
    
    # 입력 변수 선택
    st.title("Input 변수 선택")
    cols = st.multiselect("복수의 컬럼을 선택하세요.", df.columns[1:])
    st.dataframe(df[cols])

    # 계산 버튼이 눌렸을 때만 실행
    if calculation_button:
        Xt, Xts, yt, yts = train_test_split(df[cols], df[col], test_size=0.2, shuffle=False)
        
        xgb = XGBRegressor(
            max_depth=m_d,
            n_estimators=n_e,
            learning_rate=l_r,
            subsample=sb,
            random_state=2,
            n_jobs=-1
        )
        
        # 모델 훈련
        xgb.fit(Xt, yt)

        # 예측
        yt_pred = xgb.predict(Xt)
        yts_pred = xgb.predict(Xts)

        # 성능 평가
        mse_train = mean_squared_error(yt, yt_pred)
        mse_test = mean_squared_error(yts, yts_pred)
        st.markdown(f"학습 데이터 MSE: {mse_train}")
        st.markdown(f"테스트 데이터 MSE: {mse_test}")

        r2_train = r2_score(yt, yt_pred)
        r2_test = r2_score(yts, yts_pred)
        st.markdown(f"학습 데이터 R2: {r2_train}")
        st.markdown(f"테스트 데이터 R2: {r2_test}")

        # 모델 시각화
        x_var = cols[0] if cols else None  # 선택된 변수 중 첫 번째 변수 사용

        if x_var is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 학습 데이터 그래프
            ax = axes[0]
            ax.scatter(Xt[x_var], yt, s=3, label="학습 데이터 (실제)")
            ax.scatter(Xt[x_var], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
            ax.grid()
            ax.legend(fontsize=13)
            ax.set_xlabel(x_var)
            ax.set_ylabel(col)
            ax.set_title(
                rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
                fontsize=16,
            )

            # 테스트 데이터 그래프
            ax = axes[1]
            ax.scatter(Xts[x_var], yts, s=3, label="테스트 데이터 (실제)")
            ax.scatter(Xts[x_var], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
            ax.grid()
            ax.legend(fontsize=13)
            ax.set_xlabel(x_var)
            ax.set_ylabel(col)
            ax.set_title(
                rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
                fontsize=16,
            )

            # 그래프 출력
            st.pyplot(fig)
        else:
            st.warning("입력 변수를 하나 이상 선택해야 그래프를 그릴 수 있어요!")

# 앱 실행
if __name__ == "__main__":
    main()
