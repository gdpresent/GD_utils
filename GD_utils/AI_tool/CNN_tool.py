import numpy as np
import pandas as pd
import os
import time
import torch
import h5py
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset


def save_as_pd_parquet(location, pandas_df_form):
    start = time.time()
    pandas_df_form.to_parquet(f'{location}')
    print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')
def read_pd_parquet(location):
    start = time.time()
    read = pd.read_parquet(location)
    print(f'Loading Complete({round((time.time() - start) / 60, 2)}min): {location}')
    return read


def process_params(code, year_price, ExPost_return, from_day_type_int, to_day_type_int, image_size_dict, only_today=False):
    # code = 'A0015B0'
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
        # input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx]
        if (idx - from_day_type_int*2)<0:
            input_df = price_a_stock.copy()
        else:
            input_df = price_a_stock.iloc[idx - from_day_type_int*2:idx]
        if len(input_df.dropna(how='all', axis=0))==0:
            continue
        end_dt = input_df.index[-1]
        try:
            label = int(ExPost_ret_code.loc[end_dt] * 100_00_00) / 100_00_00
        except:
            label=-1
        input_df = input_df.reindex(['open', 'high', 'low', 'close', 'volume', 'C_MA', 'O_MA'], axis=1)
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

# Train
def loss_epoch(model, DL, criterion, DEVICE, optimizer = None):
    N = len(DL.dataset) # the number of data(train data일수도있고, validation data일수도 있고)
    rloss = 0; rcorrect = 0
    for x_batch, y_batch in tqdm(DL):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # inference
        y_hat = model(x_batch)

        # loss
        loss = criterion(y_hat, y_batch)

        # update -> test를 흉내내는 validation으로는 update가 되면 안 됨!
        # optimizer가 없으면 하지 말아라~
        if optimizer is not None:
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update

        # loss accumulation
        loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 마지막 18개도 32개를 곱하니까..
        rloss += loss_b # running loss

        # accuracy accumulation
        pred = y_hat.argmax(dim=1)
        corrects_b = torch.sum(pred == y_batch).item()
        rcorrect += corrects_b

    loss_e = rloss/ N # epoch loss
    accuracy_e = rcorrect/N*100
    return loss_e, accuracy_e, rcorrect
def Train_Nepoch_ES(model, train_DL, val_DL, criterion, DEVICE, optimizer,EPOCH, BATCH_SIZE, TRAIN_RATIO,save_model_path,save_history_path, N_EPOCH_ES, init_val_loss=100E100, **kwargs):
    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size=kwargs["LR_STEP"], gamma=kwargs["LR_GAMMA"])
    else:
        scheduler = None

    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}  # 나중에 그냥 train error 줄면서 accuracy는 어떻게 되는지 그런거 함 봐보자~

    # init_val_loss=100E100 # 그냥 큰 수로 initialize
    print(f'Initial Validation Loss: {init_val_loss}')
    no_improve_count = 0  # 연속으로 개선되지 않은 epoch의 횟수

    for ep in range(EPOCH):
        if ep == 0:
            print('No Update')
            torch.save(model.state_dict(), save_model_path)
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep + 1}, current_LR = {current_lr}\n")

        model.train()
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, DEVICE, optimizer)
        loss_history["train"] += [train_loss]
        acc_history["train"] += [train_acc]

        model.eval()  # test mode로 전환(validation은 test mode on 해야지~)
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion, DEVICE)
            loss_history["val"] += [val_loss]
            acc_history["val"] += [val_acc]

            if val_loss < init_val_loss:
                init_val_loss = val_loss
                print(f'reset: BEST Error - {init_val_loss}')
                no_improve_count = 0  # reset
                torch.save(model.state_dict(), save_model_path)
            else:
                # print(f'reset: BEST Error - {best_val_loss}')
                no_improve_count += 1  # increment

        if "LR_STEP" in kwargs:
            scheduler.step()
        # print loss
        print(
              f"\ntrain loss: {round(train_loss, 5)}, "
              f"val loss: {round(val_loss, 5)} \n"
              f"train acc: {round(train_acc, 1)} %, "
              f"val acc: {round(val_acc, 1)} %, time: {round(time.time() - epoch_start)} s")
        print("-" * 20)

        # Check for early stopping condition
        if no_improve_count >= N_EPOCH_ES:
            print("Early stopping triggered")
            break
            # torch.save({"model": model,
            #             "ep": ep,
            #             "optimizer": optimizer,
            #             "scheduler": scheduler}, save_model_path)

    torch.save({"loss_history": loss_history,
                "acc_history": acc_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE,
                "TRAIN_RATIO": TRAIN_RATIO}, save_history_path)

    return acc_history['train'][-1],acc_history['val'][-1], len(acc_history['train'])

# Test
def eval_loop(dataloader, net, loss_fn, DEVICE):
    # dataloader, net, loss_fn, DEVICE=test_DL, model_V2_TmPlng_body2, criterion, DEVICE
    running_loss = 0.0
    current = 0
    net.eval()
    predict = []
    codes=[]
    dates=[]
    returns=[]
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
    return avg_loss, torch.cat(predict),codes, dates,returns,targets


def inference_result_save(pred1, test_code, test_date, test_return, test_label, epoches):
    return pd.DataFrame(
        {
         "Prob_Positive": pred1,
         "종목코드": test_code,
         "return": test_return,
         "label": test_label,
         "epoch": epoches
         }, index=pd.to_datetime(test_date)).rename_axis("date").sort_index()
def get_inference_result(model_pth, history_pth, test_DL, criterion, device, model_name='05d'):
    if model_name == '05d':
        model_trained, history, Tacc, Vacc, Teps = get_d05_model_and_history(model_pth, history_pth, device)
    elif model_name == '20d':
        model_trained, history, Tacc, Vacc, Teps = get_d20_model_and_history(model_pth, history_pth, device)
    else:
        raise ValueError('model_name error')
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
class CustomDataset_today_inference_yearly(Dataset):
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


def labeling_v1(lb_tmp):
    if lb_tmp>0:
        label = 1
    else:
        label = 0
    return label
class CustomDataset_all(Dataset):
    def __init__(self, image_data_path, data_source, train, data_date,stt_date=None, until_date=None, transform=None,Pred_Hrz = 20,cap_criterion=0.2,F_day_type=20, T_day_type=20):
        self.transform = transform
        self.train=train

        if "KR" in image_data_path: self.country ="KR"
        else:self.country ="US"

        self.data_source = data_source
        self.data_date = data_date
        self.excel_path = f'./data/{self.data_source}/excel'
        self.DB_path = f'./data/{self.data_source}/DB/{self.data_date}'

        until_date = pd.to_datetime(until_date)
        until_year = until_date.year
        stt_date = pd.to_datetime(stt_date)

        if stt_date == None:
            stt_year = 1990
        else:
            stt_year = stt_date.year
        years = [x for x in range(until_year+1) if x>=stt_year]

        ExPost = self.get_ExPost_return(n_day=Pred_Hrz)
        self.last_date = ExPost.index[-1]
        date_code_list_MiniCap_remove = self.get_date_code_list_MiniCap_remove(until_date, stt_date,cap_criterion=cap_criterion)

        self.data = []
        self.codes = []
        self.dates = []
        self.returns = []
        self.labels = []
        for year in tqdm(years[::-1], desc='### Data Loading ###'):
            if not os.path.exists(f"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5"):
                print(f'파일 없음:\n\t"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5"')
                continue
            with h5py.File(f"{image_data_path}/{F_day_type}day_to_{T_day_type}day_{year}.h5", 'r') as hf:
                loaded_images = hf['images'][:]
                loaded_codes = [s.decode('utf-8') for s in hf['codes'][:]]
                loaded_dates = [s.decode('utf-8') for s in hf['dates'][:]]
                for img, code, date in zip(loaded_images, loaded_codes, loaded_dates):
                    if f'{date}_{code}' in date_code_list_MiniCap_remove:
                        ret = ExPost.loc[pd.to_datetime(date), code]
                        lab = labeling_v1(ret)

                        self.data.append(img)
                        self.codes.append(code)
                        self.dates.append(date)
                        self.returns.append(ret)
                        self.labels.append(lab)
        if self.train:
            # clear memory
            self.codes = []
            self.dates = []
            self.returns = []
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.train:
            return img, label
        else:
            code = self.codes[idx]
            date = self.dates[idx]
            returns = self.returns[idx]
            return img, code, date, returns, label
    def get_date_code_list_MiniCap_remove(self, until_date,stt_date,cap_criterion):
        # if os.path.exists(f'./data/{data_source}/DB/{data_date}/{COUNTRY}_mktcap.hd5'):
        if os.path.exists(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5'):
            # Cap_df_raw = read_pd_parquet(f'./data/{data_source}/DB/{data_date}/{COUNTRY}_mktcap.hd5')#.loc[stt_date:f'{until_date}']
            Cap_df_raw = read_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5')#.loc[stt_date:f'{until_date}']
        else:
            # Cap_df_raw = data_preprocessing(pd.read_excel(f'./data/{data_source}/{COUNTRY}_daily_data_{data_date}.xlsx', sheet_name='시가총액'))
            # save_as_pd_parquet(f'./data/{data_source}/{COUNTRY}_MktCap_{data_date}.hd5', Cap_df_raw)
            raise ValueError('데이터 없음')
        Cap_df_raw = Cap_df_raw.loc[:f'{self.last_date}']
        Cap_df_raw = Cap_df_raw.loc[stt_date:f'{until_date}']

        Cap_df_removed_20th = Cap_df_raw[Cap_df_raw.rank(ascending=True, pct=True, axis=1) >= cap_criterion]
        # print(f'Cap_df:\n{Cap_df_removed_20th}')

        dt_code_set = set()
        for dt, code in tqdm(Cap_df_removed_20th.stack().index, desc=f'시가총액 하위 {int(cap_criterion * 100)}% 종목 제거 리스트 산출'):
            dt_code_set.add(f'{dt.strftime("%Y%m%d")}_{code}')
        return dt_code_set
    def get_ExPost_return(self,n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.pct_change((n_day*2)-1, fill_method=None).shift(-(n_day*2)).dropna(how='all', axis=0).loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5')
        return output




class OHLCV_cls:
    def __init__(self, data_date, country, data_source):
        self.country = country
        self.data_source = data_source
        self.data_date = data_date
        self.excel_path = f'./data/{self.data_source}/excel'
        self.DB_path = f'./data/{self.data_source}/DB/{self.data_date}'
        os.makedirs(self.DB_path, exist_ok=True)

        print(f'{self.country} 데이터')
        print(f'데이터 기준 날짜: {self.data_date}')


        self.open, self.high, self.low, self.close, self.volume, self.code_to_name = self.get_daily_OHLCV()
        self.open_to_close = self.gen_Open_Close_concat()

        self.ExPost_dict = {
                            5: self.get_ExPost_return(n_day=5),
                            20: self.get_ExPost_return(n_day=20),
                            60: self.get_ExPost_return(n_day=60)
                            }
        self.close_MA_dict = {
                              5: self.get_Close_MA_n(n_day=5),
                              20:self.get_Close_MA_n(n_day=20),
                              60:self.get_Close_MA_n(n_day=60)
                             }
        self.open_MA_dict = {
                              5:self.get_Open_MA_n(n_day=5),
                              20:self.get_Open_MA_n(n_day=20),
                              60:self.get_Open_MA_n(n_day=60)
                             }

    # 가격 원본 데이터 Query
    def get_daily_OHLCV_KR_raw(self):
        if not os.path.exists(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5'):
            tmp_read=pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정주가')
            _code_to_name = tmp_read.iloc[6:8].T.iloc[1:].set_index(6).rename_axis('code').rename(columns={7:'name'})
            _close = self.QuantiWise_data_preprocessing(tmp_read);print('close read')
            _open = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정시가'));print('open read')
            _high = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정고가'));print('high read')
            _low = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정저가'));print('low read')
            _volume = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='수정거래량'));print('volume read')
            _mktcap = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='시가총액'));print('volume read')
            _SuspTrnsctn = self.QuantiWise_data_preprocessing(pd.read_excel(f'{self.excel_path}/{self.country}_daily_data_{self.data_date}.xlsx', sheet_name='거래정지'));print('거래정지 read')

            save_as_pd_parquet(f'{self.DB_path}/{self.country}_code_to_name_{self.data_date}.hd5', _code_to_name)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_open_{self.data_date}.hd5', _open)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_high_{self.data_date}.hd5', _high)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_low_{self.data_date}.hd5', _low)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_volume_{self.data_date}.hd5', _volume)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_close_{self.data_date}.hd5', _close)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5', _mktcap)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_SuspTrnsctn_{self.data_date}.hd5', _SuspTrnsctn)
        else:
            _open = read_pd_parquet(f'{self.DB_path}/{self.country}_open_{self.data_date}.hd5')
            _high = read_pd_parquet(f'{self.DB_path}/{self.country}_high_{self.data_date}.hd5')
            _low = read_pd_parquet(f'{self.DB_path}/{self.country}_low_{self.data_date}.hd5')
            _volume = read_pd_parquet(f'{self.DB_path}/{self.country}_volume_{self.data_date}.hd5')
            _close = read_pd_parquet(f'{self.DB_path}/{self.country}_close_{self.data_date}.hd5')
            _mktcap = read_pd_parquet(f'{self.DB_path}/{self.country}_mktcap_{self.data_date}.hd5')
            _code_to_name = read_pd_parquet(f'{self.DB_path}/{self.country}_code_to_name_{self.data_date}.hd5')
            _SuspTrnsctn = read_pd_parquet(f'{self.DB_path}/{self.country}_SuspTrnsctn_{self.data_date}.hd5')
        _open = _open[_open>0]
        _high = _high[_high>0]
        _low = _low[_low>0]
        _volume = _volume[_volume>0]
        _close = _close[_close>0]

        close = _close.dropna(how='all', axis=0).dropna(how='all', axis=1)
        open = _open.dropna(how='all', axis=0).dropna(how='all', axis=1)
        high = _high.dropna(how='all', axis=0).dropna(how='all', axis=1)
        low = _low.dropna(how='all', axis=0).dropna(how='all', axis=1)
        volume = _volume.dropna(how='all', axis=0).dropna(how='all', axis=1)

        init_dt = np.nanmax([close.index.min(), open.index.min(), high.index.min(), low.index.min(), volume.index.min()])
        last_dt = np.nanmin([close.index.max(), open.index.max(), high.index.max(), low.index.max(), volume.index.max()])

        close = _close.loc[init_dt:last_dt]
        open = _open.loc[init_dt:last_dt]
        high = _high.loc[init_dt:last_dt]
        low = _low.loc[init_dt:last_dt]
        volume = _volume.loc[init_dt:last_dt]

        return open, high, low, close, volume, _code_to_name,_SuspTrnsctn

    # 가격 데이터 전처리
    def get_daily_OHLCV(self):
        open_raw, high_raw, low_raw, close_raw, volume_raw, code_to_name, SuspTrnsctn = self.get_daily_OHLCV_KR_raw()
        open_raw[SuspTrnsctn==1]=np.nan
        high_raw[SuspTrnsctn==1]=np.nan
        low_raw[SuspTrnsctn==1]=np.nan
        close_raw[SuspTrnsctn==1]=np.nan
        volume_raw[SuspTrnsctn==1]=np.nan
        # close_raw['A476470']
        if not os.path.exists(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5'):
            # 거래량 0 -> 거래량 NaN값 처리
            volume_0_mask = volume_raw.le(0)
            # volume_raw.le(0).equals(volume_raw.eq(0))
            volume_raw[volume_0_mask] = np.nan

            # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
            # 맨 뒤에서
            # open_null_mask = open_raw.isnull()
            # close_raw[open_null_mask]=np.nan
            # high_raw[open_null_mask]=np.nan
            # low_raw[open_null_mask]=np.nan
            # volume_raw[open_null_mask]=np.nan

            # close값 없음 -> 모든 데이터 NaN값 처리(추론 불가능)
            close_null_mask = close_raw.isnull()
            open_raw[close_null_mask] = np.nan
            high_raw[close_null_mask] = np.nan
            low_raw[close_null_mask] = np.nan
            volume_raw[close_null_mask] = np.nan

            # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
            high_null_mask = high_raw.isnull()
            low_null_mask = low_raw.isnull()
            open_null_mask = open_raw.isnull()
            price_all_null_mask = open_null_mask & close_null_mask & high_null_mask & low_null_mask
            volume_raw[price_all_null_mask] = np.nan

            # 비어있는 high와 low는, 존재하는 open과 close로 filling
            # 이거는 여기에서 처리하면 안되겠다
            # --> 이미지 데이터 뽑아내기 직전에 처리했음(def process_code_idx)

            # low가 high보다 높으면 그날 모든 데이터 없다 처리(data error로 간주)
            low_exist_mask = low_raw.notna()
            high_exist_mask = high_raw.notna()
            low_exist = low_raw[low_exist_mask & high_exist_mask]
            high_exist = high_raw[low_exist_mask & high_exist_mask]
            Low_greater_than_high_mask = low_exist.gt(high_exist)
            open_raw[Low_greater_than_high_mask] = np.nan
            high_raw[Low_greater_than_high_mask] = np.nan
            low_raw[Low_greater_than_high_mask] = np.nan
            close_raw[Low_greater_than_high_mask] = np.nan
            volume_raw[Low_greater_than_high_mask] = np.nan

            # open값 없음 -> 오늘 종가/거래량, 어제 종가/거래량 존재하면 오늘의 open은 어제의 종가로 채워주자
            # open_null_mask 위에서 이미 한 번 선언 했었으니까
            today_close_exist = close_raw.notna()
            yesterday_close_exist = close_raw.notna().shift(1)
            today_open_null = open_raw.isnull()
            to_replace_value = close_raw.shift(1)[today_open_null & today_close_exist & yesterday_close_exist].stack()
            for idx in tqdm(to_replace_value.index, desc='######## NaN open filling ########'):
                open_raw.loc[idx[0], idx[1]] = to_replace_value.loc[idx]

            # 거래량 NaN -> 모든 가격 NaN값 처리
            volume_nan_mask = volume_raw.isnull()
            open_raw[volume_nan_mask] = np.nan
            high_raw[volume_nan_mask] = np.nan
            low_raw[volume_nan_mask] = np.nan
            close_raw[volume_nan_mask] = np.nan

            # 없다 처리 된 것들 있으니까 한 번 더
            # 시고저종 모두 NaN값 -> 거래량 NaN값 처리
            high_null_mask2 = high_raw.isnull()
            low_null_mask2 = low_raw.isnull()
            open_null_mask2 = open_raw.isnull()
            close_null_mask2 = close_raw.isnull()
            price_all_null_mask2 = open_null_mask2 & close_null_mask2 & high_null_mask2 & low_null_mask2
            volume_raw[price_all_null_mask2] = np.nan

            save_as_pd_parquet(f'{self.DB_path}/{self.country}_open_processed_{self.data_date}.hd5', open_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_high_processed_{self.data_date}.hd5', high_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_low_processed_{self.data_date}.hd5', low_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_volume_processed_{self.data_date}.hd5', volume_raw)
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5', close_raw)

        open_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_open_processed_{self.data_date}.hd5')
        high_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_high_processed_{self.data_date}.hd5')
        low_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_low_processed_{self.data_date}.hd5')
        volume_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_volume_processed_{self.data_date}.hd5')
        close_processed = read_pd_parquet(f'{self.DB_path}/{self.country}_close_processed_{self.data_date}.hd5')

        return open_processed, high_processed, low_processed, close_processed, volume_processed, code_to_name

    # 사후수익률 선계산
    def get_ExPost_return(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.pct_change((n_day*2)-1, fill_method=None).shift(-(n_day*2)).dropna(how='all', axis=0).loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_ExPost_return_{n_day}_{self.data_date}.hd5')
        return output

    # 시가-종가 연결
    def gen_Open_Close_concat(self):
        open_time = pd.DataFrame(self.open.values, columns=self.open.columns, index=[x+pd.DateOffset(hours=9) for x in self.open.index])
        close_time = pd.DataFrame(self.close.values, columns=self.close.columns, index=[x+pd.DateOffset(hours=16) for x in self.close.index])
        return pd.concat([open_time, close_time], axis=0).sort_index()

    # 시가-종가 연결하여 이동평균선 종가 부분을 만들어 놓음
    def get_Close_MA_n(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5'):
            tmp=self.open_to_close.rolling(min_periods=n_day*2, window=n_day*2).mean().loc[lambda x:x.index.hour==16]
            output=pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_C_MA_{n_day}_{self.data_date}.hd5')
        return output
    # 시가-종가 연결하여 이동평균선 시가 부분을 만들어 놓음
    def get_Open_MA_n(self, n_day):
        if not os.path.exists(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5'):
            tmp = self.open_to_close.rolling(min_periods=n_day * 2, window=n_day * 2).mean().loc[lambda x: x.index.hour == 9]
            output = pd.DataFrame(tmp.values, columns=tmp.columns, index=[pd.to_datetime(x) for x in tmp.index.date])
            # output_old = self.close.rolling(min_periods=n_day, window=n_day).mean()
            save_as_pd_parquet(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5', output)
        else:
            output = read_pd_parquet(f'{self.DB_path}/{self.country}_O_MA_{n_day}_{self.data_date}.hd5')
        return output

    # 1년단위로 데이터 불러오는 부분(주가데이터의 이미지화는 1년 단위로 처리함)
    def get_period_price_and_ExPostRet_v2(self, year_str, day_type):
        try:
            day_type_ago_date = self.volume.loc[f'{int(year_str) - 1}'].iloc[-int(day_type)*2:].index[0]
        except:
            day_type_ago_date = self.volume.loc[f'{int(year_str)}'].index[0]

        # 이제 모든 DataFrame을 병합합니다.
        year_price = pd.concat({
            'open': self.open.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'high': self.high.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'low': self.low.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'close': self.close.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'volume': self.volume.loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'C_MA': self.close_MA_dict[day_type].loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
            'O_MA': self.open_MA_dict[day_type].loc[day_type_ago_date:year_str].dropna(how='all', axis=1),
        }, axis=1)


        ExPost_return = self.ExPost_dict[day_type].loc[day_type_ago_date:year_str]
        return year_price, ExPost_return
    def get_period_price_and_ExPostRet_ETFonly(self, year_str, day_type, ETF_univ):
        try:
            day_type_ago_date = self.volume.loc[f'{int(year_str) - 1}'].iloc[-int(day_type)*2:].index[0]
        except:
            day_type_ago_date = self.volume.loc[f'{int(year_str)}'].index[0]

        # 이제 모든 DataFrame을 병합합니다.
        year_price = pd.concat({
            'open': self.open.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'high': self.high.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'low': self.low.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'close': self.close.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'volume': self.volume.loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'C_MA': self.close_MA_dict[day_type].loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
            'O_MA': self.open_MA_dict[day_type].loc[day_type_ago_date:year_str,ETF_univ].dropna(how='all', axis=1),
        }, axis=1)


        ExPost_return = self.ExPost_dict[day_type].loc[day_type_ago_date:year_str]
        return year_price, ExPost_return

    # Quantiwise 와꾸 데이터전처리
    def QuantiWise_data_preprocessing(self, data, univ=[]):
        data.columns = data.iloc[6]
        data = data.drop(range(0, 13), axis=0)
        data = data.rename(columns={'Code': 'date'}).rename_axis("종목코드", axis="columns").set_index('date')
        data.index = pd.to_datetime(data.index)
        if len(univ) != 0:
            data = data[univ]
        return data
    def Refinitiv_data_preprocessing(self, pvt_tmp):
        # pvt_tmp=o_pvt_raw.copy()
        pvt_tmp.index = pd.to_datetime(pvt_tmp.index)
        pvt_tmp = pvt_tmp.drop(pvt_tmp.iloc[0][pvt_tmp.iloc[0].apply(lambda x: type(x)==str)].index, axis=1)
        pvt_tmp = pvt_tmp[pvt_tmp.columns[~pvt_tmp.columns.isna()]]
        pvt_tmp = pvt_tmp.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return pvt_tmp
