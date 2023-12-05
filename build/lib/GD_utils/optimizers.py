import scipy.optimize as sco
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt
import copy
from math import *

# AE 는 기본적으로 주성분 분석(PCA)으로 처리하는 일차원 데이터 처리 방식을 딥러닝 방식으로 확장한 것
# 입력 데이터의 특징을 효율적으로 찾는 것이 목적 (저차원화를 통한 데이터 관계 관찰, 데이터 압축, 디노이징)
# 신경망을 사용하기 때문에 데이터 구성이 복잡하거나 데이터가 대량인 경우 PCA 보다 더 효과적
def portfolio(returns, weights):
    weights = np.array(weights)
    rets = returns.mean() * 252
    covs = returns.cov() * 252

    P_ret = np.sum(rets * weights) #MAX RETURN
    P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights))) #MIN VARIANCE
    P_sharpe = P_ret / P_vol #MAX SHARPE
    sqr_downside = np.square(np.clip(returns, np.NINF, 0))
    downside_risk = np.sqrt(np.nanmean(sqr_downside) * 252)

    P_sortino = P_ret / downside_risk #MAX SORTINO


    return np.array([P_ret, P_vol, P_sharpe, P_sortino])

class AutoencoderAgent:

    def __init__(
            self,
            portfolio_size,
            allow_short=True,
            encoding_dim=25
    ):

        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.encoding_dim = encoding_dim

    # 오토인코더가 선형 활성화 함수만 사용하고 비용 함수가 평균 제곱 오차(MSE) 라면, 이는 결국 주성분 분석(PCA)을 수행한다고 볼 수 있음
    # (이 소스에서는 인코딩 시 relu 사용)
    # 핸즈온 머신러닝(p524)
    def model(self):
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras import regularizers
        from keras.models import load_model
        input_img = Input(shape=(self.portfolio_size,))
        # Dense 는 완전 연결 NN 레이어 (Fully Connected Neural Network)
        encoded = Dense(self.encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(input_img)
        decoded = Dense(self.portfolio_size, activation='linear', kernel_regularizer=regularizers.l2(1e-6))(encoded)
        autoencoder = Model(input_img, decoded)
        # 신경망 성능의 '나쁨'을 나타내는 손실함수로 MSE 사용
        # MSE (Mean Squared Error) : 평균제곱오차. 추측값에 대한 정확성 측정 방법
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def act(self, returns):
        data = returns
        autoencoder = self.model()

        # 입력과 출력 모두 data 로 설정하여 학습
        # 총 25회 학습, 1회 배치마다 데이터 32개를 프로세스에 보냄
        # verbose 는 학습 중 출력문구 표시
        # shuffle 은 각 epoch 마다 샘플 섞을 지 여부
        autoencoder.fit(data, data, shuffle=False, epochs=25, batch_size=32, verbose=False)
        reconstruct = autoencoder.predict(data)

        communal_information = []

        for i in range(0, len(returns.columns)):
            # 노름(Norm) 은 벡터의 길이 혹은 크기를 측정하는 방법(함수) (http://taewan.kim/post/norm/)
            # numpy.linalg.norm 의 반환값:  매트릭스 또는 벡터의 Norm
            diff = np.linalg.norm((returns.iloc[:, i] - reconstruct[:, i]))  # 2 norm difference
            communal_information.append(float(diff))

        optimal_weights = np.array(communal_information) / sum(communal_information)

        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


# statsmodels 을 사용한 Holt Smoothing Forecast (Linear Trend)
# 모델 유형별 표 : https://ko.logpresso.com/documents/time-series
# 기본적인 예측 보정 과정 : https://m.blog.naver.com/kiddwannabe/220050965511
class SmoothingAgent:

    def __init__(
            self,
            portfolio_size,
            allow_short=True,
            forecast_horizon=252,
    ):

        self.portfolio_size = portfolio_size  # 자산 갯수
        self.allow_short = allow_short
        self.forecast_horizon = forecast_horizon  # 홀딩 기간

    # 자산 별 비중 리스트 반환하는 함수
    def act(self, timeseries):

        optimal_weights = []

        for asset in timeseries.columns:
            ts = timeseries[asset]
            # Holt's Linear Trend 예측을 사용
            # (다른 방법으로는 Exponential trend, Additive damped trend 등이 있음)
            fit1 = Holt(ts).fit()
            # 홀딩 기간만큼의 미래를 예측
            forecast = fit1.forecast(self.forecast_horizon)
            # 예측값의 크기를 비중값으로 사용
            prediction = forecast.values[-1] - forecast.values[0]
            optimal_weights.append(prediction)

        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 가장 작은 수치의 절대값을 더해서 음수가 나오지 않도록 함
            # (음수 비중의 자산이 없을 땐 비중의 편차가 줄어드는 문제(?) )

            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


# Principal Component Analysis (주성분 분석)
# https://systematicedge.wordpress.com/2013/06/02/principal-component-analysis-in-portfolio-management/
# (리스크 기반의 전통적 자산배분은 채권 비중이 높아진다는 걸 언급)
class PCAAgent:
    def __init__(
            self,
            portfolio_size,
            pc_id=0,
            pca_max=10,  # = 10,  #자산 8개 시 오류로 수정해 봄
            allow_short=False,
    ):

        self.portfolio_size = portfolio_size  # 자산 갯수
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.pc_id = pc_id
        self.pc_max = pca_max

    def act(self, returns):
        C = self.pc_max  # 유지할 컴포넌트 수
        # pca = PCA(C) #pc_max 가 n_components 로 설정됨. 값이 없으면 모든 컴포넌트가 유지
        pca = PCA()
        returns_pca = pca.fit_transform(returns)  # 특징행렬을 낮은 자원의 근사행렬로 변환
        pcs = pca.components_  # ETF 가격을 다르게 하는 요인(주성분)

        pc1 = pcs[self.pc_id, :]

        if self.allow_short:
            optimal_weights = pc1 / sum(np.abs(pc1))
        else:
            # 오류 해결을 위해 추가
            # optimal_weights = pc1 / sum(np.abs(pc1)) #테스트1
            optimal_weights = pc1  # 테스트2
            # optimal_weights = []

            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)
        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)
        return optimal_weights


class MaxReturnsAgent:
    def __init__(
            self,
            portfolio_size,
            ub=1.,
            lb=0.05,
            allow_short=False,

    ):

        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.ub = ub
        self.lb = lb

    def act(self, returns):

        # 최적화 목적함수
        def loss(weights):
            return -portfolio(returns, weights)[0]

        n_assets = len(returns.columns)

        if self.allow_short:
            bnds = tuple((-1.0, 1.0) for x in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            bnds = tuple((self.lb, self.ub) for x in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

        opt_S = sco.minimize(
            loss,
            n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bnds,
            constraints=cons)

        optimal_weights = opt_S['x']

        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


class MinVarianceAgent:
    def __init__(
            self,
            portfolio_size,
            ub=1.,
            lb=0.05,
            allow_short=False,
    ):

        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.ub = ub
        self.lb = lb

    def act(self, returns):

        def loss(weights):
            return portfolio(returns, weights)[1] ** 2

        n_assets = len(returns.columns)

        if self.allow_short:
            bnds = tuple((-1.0, 1.0) for x in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            bnds = tuple((self.lb, self.ub) for x in range(n_assets))
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

        opt_S = sco.minimize(
            loss,
            n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bnds,
            constraints=cons)

        optimal_weights = opt_S['x']

        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


class MaxSharpeAgent:

    def __init__(
            self,
            portfolio_size,
            ub=1.,
            lb=0.05,
            allow_short=False,
    ):

        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size)
        self.ub = ub
        self.lb = lb

    def act(self, returns):

        def loss(weights):
            return -portfolio(returns, weights)[2]

        n_assets = len(returns.columns)

        if self.allow_short:
            # x 범위 (-1 ~ 1)
            bnds = tuple((-1.0, 1.0) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            # x 범위 (0 ~ 1)
            bnds = tuple((self.lb, self.ub) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

        opt_S = sco.minimize(
            loss,
            n_assets * [1.0 / n_assets],  # 초깃값 벡터
            method='SLSQP',
            # bounds 와 constraints: 구속최적화(constrained optimization) 문제에서의 구속조건을 부과하는데 사용 (https://wikidocs.net/15656)
            bounds=bnds,  # short 허용 여부에 따라 x 범위 지정
            constraints=cons  # 제약 조건 'eq' - equality , fun - constraint 함수
        )

        optimal_weights = opt_S['x']

        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


class MaxDecorrelationAgent:

    def __init__(
            self,
            portfolio_size,
            ub=1.,
            lb=0.05,
            allow_short=False,
    ):

        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.ub = ub
        self.lb = lb

    def act(self, returns):

        # 최적화 목적함수
        def loss(weights):
            weights = np.array(weights)
            # numpy.dot 은 두 배열의 내적곱(dot product). (numpy matmul 은 두 배열의 행렬곱(matrix product)
            return np.sqrt(np.dot(weights.T, np.dot(returns.corr(), weights)))

        n_assets = len(returns.columns)

        if self.allow_short:
            # x 범위 (-1 ~ 1)
            bnds = tuple((-1.0, 1.0) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            # x 범위 (0 ~ 1)
            bnds = tuple((self.lb, self.ub) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

        '''
        #Scipy minimize method 의 종류

        Nelder - Mead’
        ‘Powell’
        ‘CG’
        ‘BFGS’
        ‘Newton - CG’
        ‘L - BFGS - B’
        ‘TNC’
        ‘COBYLA’
        ‘SLSQP’
        ‘trust - constr’
        ‘dogleg’
        ‘trust - ncg’
        ‘trust - exact’
        ‘trust - krylov’
        '''
        opt_S = sco.minimize(
            loss,
            n_assets * [1.0 / n_assets],  # 초깃값 벡터
            method='SLSQP',
            # bounds 와 constraints: 구속최적화(constrained optimization) 문제에서의 구속조건을 부과하는데 사용 (https://wikidocs.net/15656)
            bounds=bnds,  # short 허용 여부에 따라 x 범위 지정
            constraints=cons  # 제약 조건 'eq' - equality , fun - constraint 함수
        )

        optimal_weights = opt_S['x']

        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 가장 작은 수치의 절대값을 더해서 음수가 나오지 않도록 함
            # (음수 비중의 자산이 없을 땐 비중의 편차가 줄어드는 문제(?) )
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


class MaxDiversificationAgent:
    def __init__(self,
                 portfolio_size,
                 ub=1.,
                 lb=0.05,
                 allow_short=False, ):
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.ub = ub
        self.lb = lb

    def act(self, returns):
        data = returns
        '''
        lb = 0.1
        ub = 0.5
        '''
        Target = 0.05

        cov = data.cov()
        mean = ((1 + data.mean()) ** 12) - 1

        def minvar_objective(x):
            variance = x.T @ cov @ x
            sigma = (variance ** 0.5) * np.sqrt(12)
            mean_return = x.T @ mean
            shape = -(mean_return / sigma)

            return (shape)

        def weight_sum_constraint(x):
            return (x.sum() - 1.0)

        def TargetVol_const_lower(x):
            variance = x.T @ cov @ x
            sigma = variance ** 0.5
            sigma_scale = sigma * np.sqrt(12)

            vol_diffs = 0
            # vol_diffs = sigma_scale - (Target * 0.95)

            return (vol_diffs)

        def TargetVol_const_upper(x):
            variance = x.T @ cov @ x
            sigma = variance ** 0.5
            sigma_scale = sigma * np.sqrt(12)

            vol_diffs = (Target * 1.05) - sigma_scale

            return (vol_diffs)

        def loss(weights):
            weights = np.array(weights)

            cors = returns[-21:].corr()
            stds = returns[-63:].std()
            covs = (np.diag(stds) @ cors @ np.diag(stds)) * 252
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))

            sigma_vector = np.diag(covs)
            W_vol = weights.T @ sigma_vector
            P_Div = W_vol / P_vol

            return -P_Div

        n_assets = len(returns.columns)

        if self.allow_short:
            # x 범위 (-1 ~ 1)
            bnds = tuple((-1.0, 1.0) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            # x 범위 (0 ~ 1)
            bnds = tuple((self.lb, self.ub) for x in range(n_assets))
            # 제약 조건 'eq' - equality , fun - constraint 함수
            # cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})
            cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)}, {'type': 'ineq', 'fun': TargetVol_const_lower},
                    {'type': 'ineq', 'fun': TargetVol_const_upper})

        opt_S = sco.minimize(
            loss,
            n_assets * [1.0 / n_assets],  # 초깃값 벡터
            method='SLSQP',
            # bounds 와 constraints: 구속최적화(constrained optimization) 문제에서의 구속조건을 부과하는데 사용 (https://wikidocs.net/15656)
            bounds=bnds,  # short 허용 여부에 따라 x 범위 지정
            constraints=cons  # 제약 조건 'eq' - equality , fun - constraint 함수
        )

        optimal_weights = opt_S['x']

        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            # 0보다 작은 비중이 있을 때만 보정하도록 코드 추가
            if np.min(optimal_weights) < 0:
                optimal_weights += np.abs(np.min(optimal_weights))

            optimal_weights /= sum(optimal_weights)

        # print("===============max diver=================")
        # print(optimal_weights)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights


class MinCorrelationAgent:
    def __init__(
            self,
            portfolio_size,
            method_value,
            allow_short=False,
            # 상관계수 값에 부여하는 비중 승수
            power_value=2
    ):
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        # 상관계수 값에 부여하는 비중 승수
        self.power_value = power_value
        self.method_value = method_value

    ####################################################################################################################
    # 표준정규분포 함수

    # 1.2 x 10 ^ (-7) 미만의 분수 오차를 갖는 오차 보완 함수
    # Complementary error function.
    # http://www.sicopolis.net/docu/doxygen/v30/erfcc_8F90.html
    def erfcc(self, x):
        z = abs(x)
        t = 1. / (1. + 0.5 * z)
        r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 + t * (.09678418 + t * (-.18628806 + t * (
                .27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 + t * .17087277)))))))))
        if (x >= 0.):
            return r
        else:
            return 2. - r

    # cdf = Cumulative Distribution Function (누적 분포 함수)
    # https://mathbits.com/MathBits/TISection/Statistics2/normaldistribution.htm
    def normcdf(self, x, mu, sigma):
        t = x - mu
        y = 0.5 * self.erfcc(-t / (sigma * sqrt(2.0)))
        if y > 1.0:
            y = 1.0
        return y

    # pdf = Probability Density Function  (확률 밀도 함수)
    def normpdf(self, x, mu, sigma):
        u = (x - mu) / abs(sigma)
        y = (1 / (sqrt(2 * pi) * abs(sigma))) * exp(-u * u / 2)
        return y

    # cdf/pdf 선택 함수
    def normdist(self, x, mu, sigma, f):
        if f:
            y = self.normcdf(x, mu, sigma)
        else:
            y = self.normpdf(x, mu, sigma)
        return y

    # 정규분포 구간 값 리턴 함수
    def normrange(self, x1, x2, mu, sigma, f=True):
        # Calculates probability of random variable falling between two points.
        p1 = self.normdist(x1, mu, sigma, f)
        p2 = self.normdist(x2, mu, sigma, f)
        return abs(p1 - p2)

    # weight 계산 실행
    def act(self, returns):
        etf_data_part = returns

        # simple geometric return
        temp_df = copy.deepcopy(etf_data_part)
        etf_data_return = temp_df

        # 마지막 날짜 저장
        # loop_last_date = etf_data_return['date'].iloc[-1]
        # 데이터에서 날짜 버림
        # etf_data_return.drop(['date'], axis='columns', inplace=True)

        # etf_data_return = etf_data_return.set_index('date')
        # 수익률로 변환
        # etf_data_return = etf_data_return.pct_change()
        # 수익률 계산이 불가한 첫 날 제거
        etf_data_return = etf_data_return.dropna(axis=0)
        etf_data_return = etf_data_return.reset_index(drop=True)
        # 누적 수익률 계산
        etf_data_return_cum = (1 + etf_data_return).cumprod() - 1

        # 수익률 양수로 변경 코드 - makeplus
        if etf_data_return_cum.iloc[-1].min() < 0:
            etf_data_last_cum = etf_data_return_cum.iloc[-1] - etf_data_return_cum.iloc[-1].min()
        else:
            etf_data_last_cum = etf_data_return_cum.iloc[-1]

        if self.power_value == 3:
            # max return 활용 같은 형태
            etf_data_vol = 1 / etf_data_last_cum
        elif self.power_value == 2:
            # vol/cum 을 사용 (vol 코드를 같이 이용하므로 작은 값이 좋은 형태로 입력이 되어야 함) = sharpe ratio 같은 형태
            etf_data_vol = etf_data_return.std() / etf_data_last_cum
        else:
            # 선택 자산 변동성 계산
            etf_data_vol = etf_data_return.std()

        # 선택 자산 상관계수 매트릭스
        etf_correl_matrix = etf_data_return.corr()
        # 전체가 채워진 상관계수 메트릭스 확인용 출력
        # seaborn.heatmap(etf_correl_matrix, annot=True)

        # 선택 자산 대각선 기준 한 쪽만 남은 상관계수 메트릭스
        etf_correl_matrix.values[np.tril_indices_from(etf_correl_matrix.values)] = np.nan
        # seaborn.heatmap(etf_correl_matrix, annot=True)

        # 대각선 기준 한 쪽만 남은 상관계수 매트릭스의 평균 계산
        correl_avg = etf_correl_matrix.unstack().mean()
        # 대각선 기준 한 쪽만 상관계수 매트릭스의 표준편차 계산
        correl_std = etf_correl_matrix.unstack().std()

        # 전체가 채원 진 상관계수 매트릭스
        etf_correl_matrix = etf_data_return.corr()
        # 상관계수 매트릭스를 리스트로 변경
        etf_correl_matrix_list = etf_correl_matrix.values.tolist()

        arr = np.array(etf_correl_matrix_list).astype(np.float)
        # 대각선의 자기상관계수(1) 제거
        np.fill_diagonal(arr, np.nan)
        # 대각선의 자기상관계수(1) 제거된 상관계수 매트릭스
        etf_correl_matrix = pd.DataFrame(arr, columns=etf_data_return.columns)

        # 각 자산별 타 자산과의 상관계수 평균 계산
        stock_correl_avg = pd.DataFrame(etf_correl_matrix.mean(), columns=['avg'])
        # 각 자산별 타 자산관의 상관계수 순위 계산
        stock_correl_avg['stock_rank'] = stock_correl_avg['avg'].rank(ascending=True, method='first')

        # 상관계수에 부여하는 비중 승수
        # power_value = 1

        # 상관계수 순위에 승수 부여
        stock_correl_avg['power'] = stock_correl_avg['stock_rank'] ** self.power_value
        # 승수 부여된 값의 합계
        power_sum = stock_correl_avg['power'].sum()
        # 리스케일된 자산 상관계수를 조정 (모든 자산 상관계수의 합이 1이 되게 처리)
        stock_correl_avg['RR_adj'] = stock_correl_avg['power'] / power_sum

        temp_matrix = copy.deepcopy(etf_correl_matrix)
        normdist_matrix = temp_matrix

        # 정규분포 매트릭스 생성
        for c_row in range(len(etf_correl_matrix)):
            for c_col in range(len(etf_correl_matrix)):
                normdist_matrix.iloc[c_row, c_col] = 1 - self.normdist(etf_correl_matrix.iloc[c_row, c_col], correl_avg,
                                                                       correl_std, True)

        temp_matrix2 = copy.deepcopy(normdist_matrix)
        RankWeight_normdist_matrix = temp_matrix2

        # 순위 비중 가미된 정규 분포 매트릭스 생성
        for i in range(len(normdist_matrix)):
            for j in range(len(normdist_matrix)):
                RankWeight_normdist_matrix.iloc[i, j] = normdist_matrix.iloc[i, j] * stock_correl_avg.iloc[
                    i, len(stock_correl_avg.columns) - 1]

        # stock_correl_avg['RankWeight_sum'] 컬럼 생성 (넣어준 초기 값은 의미없음)
        stock_correl_avg['RankWeight_sum'] = stock_correl_avg['power']
        # 각 자산의 순위 비중 가미된 정규 분포 값의 합계 계산
        for k in range(len(RankWeight_normdist_matrix)):
            stock_correl_avg.iloc[k, len(stock_correl_avg.columns) - 1] = RankWeight_normdist_matrix.iloc[:, k].sum()

        # 전체 자산의 순위 비중 가미된 정규 분포 값의 합계 계산
        RankWeight_sum_total = stock_correl_avg['RankWeight_sum'].sum()

        # 각 자산의 순위 비중 가미된 정규 분포 값의 조정 (자산 합계가 1이 되게)
        stock_correl_avg['Rescale_RankWeight'] = stock_correl_avg['RankWeight_sum'] / RankWeight_sum_total
        # print(stock_correl_avg['Rescale_RankWeight'].sum())

        Vol_list = [etf_data_vol]

        # stock_correl_avg['Volatility'] 컬럼 생성 (넣어준 초기 값은 의미없음)
        stock_correl_avg['Volatility'] = stock_correl_avg['power']
        # 각 자산 변동성 입력
        for m in range(len(stock_correl_avg)):
            stock_correl_avg.iloc[m, len(stock_correl_avg.columns) - 1] = Vol_list[0][m]

        # 각 자산의 역변동성 계산
        stock_correl_avg['inverse_Vol'] = 1 / stock_correl_avg['Volatility']
        # 각 자산 역변동성의 합계
        Inverse_Vol_sum_total = stock_correl_avg['inverse_Vol'].sum()

        # 각 자산 역변동성의 비중 조정 (비중 합계가 1이 되게 조정)
        stock_correl_avg['inverse_Vol_Weight'] = stock_correl_avg['inverse_Vol'] / Inverse_Vol_sum_total
        # print(stock_correl_avg['inverse_Vol_Weight'].sum())

        # 각 자산의 순위 비중 가미된 정규 분포 값의 조정값과 각 자산 역변동성의 비중 조정값의 곱 계산
        stock_correl_avg['Rescale_RankW_inverse_VolW'] = stock_correl_avg['Rescale_RankWeight'] * stock_correl_avg[
            'inverse_Vol_Weight']
        Rescale_RankW_inverse_VolW_sum_total = stock_correl_avg['Rescale_RankW_inverse_VolW'].sum()

        # 비중값을 1로 변경하여 최종 자산별 비중 계산
        stock_correl_avg['Final_Weight'] = stock_correl_avg[
                                               'Rescale_RankW_inverse_VolW'] / Rescale_RankW_inverse_VolW_sum_total

        # print(stock_correl_avg['Final_Weight'].sum(), data_start)
        # print(stock_correl_avg['Final_Weight'].sum())

        # lookback 기간별 결과 값중 최종 비중만 남겨 역행렬 처리
        asset_weight = pd.DataFrame(stock_correl_avg['Final_Weight']).T

        # 비중 NaN 값 0 으로 바꿈
        return_weight = asset_weight.fillna(0)

        # 비중 row 만 list 로 바꿈
        return_list = return_weight.iloc[0].tolist()

        # print("===============min correl=================")
        # print(return_list)

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # return_list = min_max_weight_scaler(return_list, 0.02, 0.5)

        return return_list


# Risk Parity 에이전트
class RiskParityAgent:
    def __init__(
            self,
            portfolio_size,
            ub=1.,
            lb=0.05,
            allow_short=False,
    ):
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)
        self.ub = ub
        self.lb = lb

    def act(self, returns):
        data = returns
        covmat = data.cov()

        n_assets = len(returns.columns)

        def RC(weight, covmat):
            weight = np.array(weight)
            variance = weight.T @ covmat @ weight
            sigma = variance ** 0.5
            mrc = 1 / sigma * (covmat @ weight)
            rc = weight * mrc
            rc = rc / rc.sum()
            return (rc)

        def RiskParity_objective(x):
            variance = x.T @ covmat @ x
            sigma = variance ** 0.5
            mrc = 1 / sigma * (covmat @ x)
            rc = x * mrc
            # a = np.reshape(rc, (len(rc), 1))
            a = np.reshape(rc.tolist(), (len(rc), 1))
            risk_diffs = a - a.T
            sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
            return (sum_risk_diffs_squared)

        def weight_longonly(x):
            return (x)

        def weight_sum_constraint(x):
            return (x.sum() - 1.0)

        x0 = np.repeat(1 / covmat.shape[1], covmat.shape[1])

        constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                       {'type': 'ineq', 'fun': weight_longonly})

        options = {'ftol': 1e-20, 'maxiter': 800}
        bnds = tuple((self.lb, self.ub) for x in range(n_assets))

        opt_S = sco.minimize(fun=RiskParity_objective,
                             x0=x0,
                             method='SLSQP',
                             constraints=constraints,
                             options=options)

        optimal_weights = opt_S['x']

        #####################################################################
        # 스케일러 적용 테스트
        #####################################################################
        # optimal_weights = min_max_weight_scaler(optimal_weights, 0.02, 0.5)

        return optimal_weights
