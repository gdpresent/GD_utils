import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset

def process_params(code, year_price, ExPost_return, from_day_type_int, to_day_type_int, image_size_dict, only_today=False):
    # from_day_type_int, to_day_type_int = from_day_type, to_day_type
    local_params_list = []
    ExPost_ret_code = ExPost_return[code]
    price_a_stock = year_price[year_price.columns[year_price.columns.get_level_values(-1) == code]]#.dropna(how='all', axis=0)
    price_a_stock = price_a_stock.stack(1).droplevel(1).reindex(year_price.index)

    first_date = price_a_stock.dropna(how='all', axis=0).index[0]
    last_date = price_a_stock.dropna(how='all', axis=0).index[-1]
    price_a_stock = price_a_stock.loc[first_date:]

    if only_today:
        stt_idx = len(price_a_stock.loc[first_date:last_date].index)
    else:
        stt_idx = from_day_type_int*2 + 1

    for idx in range(stt_idx, len(price_a_stock.loc[first_date:last_date].index)+1):
        input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx]
        if len(input_df.dropna(how='all', axis=0))==0:
            continue
        end_dt = input_df.index[-1]
        try:
            label = int(ExPost_ret_code.loc[end_dt] * 100_00_00) / 100_00_00
        except:
            label=-1
        input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx].reindex(['open', 'high', 'low', 'close', 'volume', 'C_MA', 'O_MA'], axis=1)
        chunk_size = from_day_type_int//to_day_type_int

        transformed_input = pd.DataFrame()
        for dt in input_df.index[chunk_size-1::chunk_size]:
            chunk_input = input_df.loc[:dt].iloc[-chunk_size:]

            o = chunk_input['open'].iloc[0]
            h = chunk_input['high'].max()
            l = chunk_input['low'].min()
            c = chunk_input['close'].iloc[-1]
            v = chunk_input['volume'].sum()
            transformed_input.loc[dt,['open','high','low','close','volume']] = [o,h,l,c,v]
        if len(transformed_input.dropna(how='all', axis=0))==0:
            continue
        open_time = pd.DataFrame(transformed_input.open.values, columns=['o_to_c'], index=[x+pd.DateOffset(hours=9) for x in transformed_input.open.index])
        close_time = pd.DataFrame(transformed_input.close.values, columns=['o_to_c'], index=[x+pd.DateOffset(hours=16) for x in transformed_input.close.index])
        open_to_close_time=pd.concat([open_time, close_time], axis=0).sort_index()
        C_MA_tmp=open_to_close_time.rolling(min_periods=to_day_type_int * 2, window=to_day_type_int * 2).mean().loc[lambda x: x.index.hour == 16]
        O_MA_tmp=open_to_close_time.rolling(min_periods=to_day_type_int * 2, window=to_day_type_int * 2).mean().loc[lambda x: x.index.hour == 9]

        C_MA_tmp.index = pd.to_datetime(C_MA_tmp.index.strftime("%Y-%m-%d"))
        O_MA_tmp.index = pd.to_datetime(O_MA_tmp.index.strftime("%Y-%m-%d"))

        transformed_input['C_MA'] =C_MA_tmp
        transformed_input['O_MA'] =O_MA_tmp

        #### 가격 전처리 들어가야하네....
        # 거래량 0 -> 거래량 NaN값 처리
        transformed_input.loc[transformed_input['volume']<=0, 'volume'] =np.nan

        # close값 없음 -> 모든 데이터 NaN값 처리(추론 불가능)
        transformed_input.loc[transformed_input['close'].isnull(), :] =np.nan

        # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
        transformed_input.loc[
            (transformed_input['open'].isnull())&
            (transformed_input['high'].isnull())&
            (transformed_input['low'].isnull())&
            (transformed_input['close'].isnull()), 'volume'] = np.nan

        # 비어있는 high와 low는, 존재하는 open과 close로 filling
        # 이거는 여기에서 처리하면 안되겠다
        # --> 이미지 데이터 뽑아내기 직전에 처리했음(def process_code_idx)

        # low가 high보다 높으면 그날 모든 데이터 없다 처리(data error로 간주)
        transformed_input.loc[transformed_input['low']>transformed_input['high'], :] =np.nan

        # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
        transformed_input.loc[
        (transformed_input['open'].isnull())&
        (transformed_input['close'].notna())&
        (transformed_input['volume'].notna())&
        (transformed_input['close'].notna().shift(1))&
        (transformed_input['volume'].notna().shift(1))
        , 'open'] = transformed_input['close'].shift(1)

        # 거래량 NaN -> 모든 가격 NaN값 처리
        transformed_input.loc[transformed_input['volume'].isnull(), :] =np.nan

        # 없다 처리 된 것들 있으니까 한 번 더
        # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
        transformed_input.loc[
            (transformed_input['open'].isnull())&
            (transformed_input['high'].isnull())&
            (transformed_input['low'].isnull())&
            (transformed_input['close'].isnull()), 'volume'] = np.nan

        transformed_input['high'] = transformed_input[['open', 'high', 'close']].max(1)
        transformed_input['low'] = transformed_input[['open', 'low', 'close']].min(1)

        transformed_input = transformed_input.iloc[-to_day_type_int:]

        if (len(transformed_input.dropna(how='all', axis=0))<to_day_type_int)or(len(transformed_input.dropna(how='all', axis=1).columns)<len(transformed_input.columns)):
            continue
        local_params_list.append((transformed_input, code, end_dt, label, to_day_type_int, image_size_dict))
    return local_params_list
def convert_ohlcv_to_image(df, height, interval):
    # 원하는 데이터 크기의 행렬을 만들어 zero(black)를 다 깔아놓고 (0:black, 255:white)
    img = np.zeros((height, 3 * interval), dtype=np.uint8)

    # Price scaling
    min_price = np.nanmin([df['close'].min(), df['open'].min(), df['low'].min(), df[f'C_MA'].min(), df[f'O_MA'].min()])
    max_price = np.nanmax([df['close'].max(), df['open'].max(), df['high'].max(), df[f'C_MA'].max(), df[f'O_MA'].max()])
    max_volume = df['volume'].max()

    # 이미지 픽셀 값을 날짜 하나씩 채워 넣는 방식
    if max_price - min_price!=0:
        price_scale = (height * 4 / 5) / (max_price - min_price)
        volume_scale = (height / 5) / max_volume
        for i, (_, row) in enumerate(df.iterrows()):
            if not np.isnan(row['open']):
                open_pixel = int(np.round((row['open'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + open_pixel), 3 * i] = 255
            if not np.isnan(row['close']):
                close_pixel = int(np.round((row['close'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + close_pixel), 3*i + 2] = 255
            if not np.isnan(row['high']) and not np.isnan(row['low']):
                high_pixel = int(np.round((row['high'] - min_price) * price_scale))
                low_pixel = int(np.round((row['low'] - min_price)* price_scale))
                img[-int((height * 1 / 5) + high_pixel):-int((height * 1 / 5) + low_pixel)+1, 3 * i + 1] = 255
            if not np.isnan(row[f'O_MA']):
                Oma_pixel = int(np.round((row[f'O_MA'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + Oma_pixel), 3*i] = 255
            if not np.isnan(row[f'C_MA']):
                Cma_pixel = int(np.round((row[f'C_MA'] - min_price) * price_scale))
                img[-int((height * 1 / 5) + Cma_pixel), 3*i+2] = 255
            if not np.isnan(row[f'C_MA']) and not np.isnan(row[f'O_MA']):
                Oma_pixel = int(np.round((row[f'O_MA'] - min_price) * price_scale))
                Cma_pixel = int(np.round((row[f'C_MA'] - min_price) * price_scale))
                ma_pixel = int((Oma_pixel+Cma_pixel)*0.5)
                img[-int((height * 1 / 5) + ma_pixel), 3*i+1] = 255
            if not np.isnan(row['volume']):
                volume_pixel = int(np.round(row['volume'] * volume_scale))
                if volume_pixel!=0:
                    img[-int(volume_pixel):, 3 * i + 1] = 255
    else:
        return None
    return img
def process_imaging(params):
    input_df, code, end_dt, label, day_type, image_size_dict = params
    img = convert_ohlcv_to_image(input_df, image_size_dict[day_type][0], image_size_dict[day_type][1])
    return img, code, end_dt.strftime("%Y%m%d"), label


def inference_result_save(pred1, test_code, test_date, test_return, test_label, epoches):
    return pd.DataFrame(
        {
         "Prob_Positive": pred1,
         "종목코드": test_code,
         "return": test_return,
         "label": test_label,
         "epoch": epoches
         }, index=pd.to_datetime(test_date)).rename_axis("date").sort_index()
def get_inference_result(model_pth, history_pth, test_DL, criterion, device):
    model_trained, history, Tacc, Vacc, Teps = get_d05_model_and_history(model_pth, history_pth, device)
    avg_loss, preds_tmp, codes, dates, returns, labels = eval_loop(test_DL, model_trained, criterion, device)
    one_preds_1 = torch.nn.Softmax(dim=1)(preds_tmp)[:, 1].cpu().numpy()
    inference_result = inference_result_save(one_preds_1, codes, dates, returns, labels, Teps)
    return inference_result

class baseline_CNN_20day(nn.Module):
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)  # underscore 추가
            m.bias.data.fill_(0.01)

    def __init__(self, dr_rate=0.5, stt_chnl=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=stt_chnl, out_channels=64, kernel_size=(5, 3), padding=(1, 1), stride=(3, 1),
                      dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1))
        self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 3), padding=(3, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(2, 1))
        self.conv2.apply(self.init_weights)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 3), padding=(2, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(2, 1))
        self.conv3.apply(self.init_weights)

        self.fc = nn.Linear(277248, 2)
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
class baseline_CNN_5day(nn.Module):
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)  # underscore 추가
            m.bias.data.fill_(0.01)

    def __init__(self, dr_rate=0.5, stt_chnl=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=stt_chnl, out_channels=64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)),
            )
        self.conv1.apply(self.init_weights)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 3), padding=(2, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d((2, 1)),
                                   )
        self.conv2.apply(self.init_weights)

        self.fc = nn.Linear(15360, 2)
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
def get_d20_model_and_history(mdl_pth, hry_pth, DEVICE):
    try:
        model = baseline_CNN_20day(dr_rate=0.5, stt_chnl=1).to(DEVICE)
        model.load_state_dict(torch.load(mdl_pth, map_location=DEVICE))
    except:
        model = nn.DataParallel(baseline_CNN_20day(dr_rate=0.5, stt_chnl=1)).to(DEVICE)
        model.load_state_dict(torch.load(mdl_pth, map_location=DEVICE))
    hstry = torch.load(hry_pth)
    Tacc, Vacc, eps = hstry['acc_history']['train'][-1], hstry['acc_history']['val'][-1], len(
        hstry['acc_history']['train'])
    return model, hstry, Tacc, Vacc, eps
def get_d05_model_and_history(mdl_pth, hry_pth, DEVICE):
    try:
        model = baseline_CNN_5day(dr_rate=0.5, stt_chnl=1).to(DEVICE)
        model.load_state_dict(torch.load(mdl_pth, map_location=DEVICE))
    except:
        model = nn.DataParallel(baseline_CNN_5day(dr_rate=0.5, stt_chnl=1)).to(DEVICE)
        model.load_state_dict(torch.load(mdl_pth, map_location=DEVICE))
    hstry = torch.load(hry_pth)
    Tacc, Vacc, eps = hstry['acc_history']['train'][-1], hstry['acc_history']['val'][-1], len(
        hstry['acc_history']['train'])
    return model, hstry, Tacc, Vacc, eps
def eval_loop(dataloader, net, loss_fn, DEVICE):
    # dataloader, net, loss_fn, DEVICE=test_DL, model_V2_TmPlng_body2, criterion, DEVICE
    # dataloader, net, loss_fn, DEVICE=test_DL_v_05to05, bCNN_v_050505_model_1, criterion, DEVICE
    running_loss = 0.0
    current = 0
    net.eval()
    predict = []
    codes = []
    dates = []
    returns = []
    target = []
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (img, code, date, rets, label) in enumerate(t):
                X = img.to(DEVICE)
                y = label.to(DEVICE)
                y_pred = net(X)
                target.append(y.detach())
                codes.extend(code)
                dates.extend(date)
                returns.append(rets.detach())
                predict.append(y_pred.detach())
                loss = loss_fn(y_pred, y.long())
                running_loss += loss.item() * len(X)  # Here we update the running_loss
                avg_loss = running_loss / (current + len(X))
                t.set_postfix({'running_loss': avg_loss})
                current += len(X)
    returns = torch.cat(returns).cpu().numpy()
    targets = torch.cat(target).cpu().numpy()
    return avg_loss, torch.cat(predict), codes, dates, returns, targets
class CustomDataset_today_inference(Dataset):
    def __init__(self, image_data_path, F_day_type=20, T_day_type=20, transform=None):
        self.transform = transform

        self.data = []
        self.codes = []
        self.dates = []
        self.returns=[]
        self.labels=[]
        with h5py.File(f"{image_data_path}/{F_day_type}day_to_{T_day_type}day.h5", 'r') as hf:
            loaded_images = hf['images'][:]
            loaded_codes = [s.decode('utf-8') for s in hf['codes'][:]]
            loaded_dates = [s.decode('utf-8') for s in hf['dates'][:]]
            for img, code, date in zip(loaded_images, loaded_codes, loaded_dates):
                self.data.append(img)
                self.codes.append(code)
                self.dates.append(date)
                self.returns.append(0)
                self.labels.append(0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        code = self.codes[idx]
        date = self.dates[idx]
        returns = self.returns[idx]
        return img, code, date, returns, label
