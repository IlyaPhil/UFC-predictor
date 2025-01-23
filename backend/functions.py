import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union, Annotated
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

# Создаем новые признаки
def create_new_features(df: pd.DataFrame) -> pd.DataFrame :
    """
    Создает новые признаков, представляющие собой разницу между значениями признаков двух бойцов
    """
    df_copy = df.copy()   
    df_copy['Age_diff'] = df_copy['RedAge'] - df_copy['BlueAge']
    df_copy['Height_diff'] = df_copy['RedHeightCms'] - df_copy['BlueHeightCms']
    df_copy['Reach_diff'] = df_copy['RedReachCms'] - df_copy['BlueReachCms']
    df_copy['Weight_diff'] = df_copy['RedWeightLbs'] - df_copy['BlueWeightLbs']   
    df_copy['WinsByKO/TKO_diff'] = df_copy['RedWinsByKO_TKO'] - df_copy['BlueWinsByKO_TKO']
    df_copy['WinsBySubmission_diff'] = df_copy['RedWinsBySubmission'] - df_copy['BlueWinsBySubmission']
    df_copy['WinsByDecision_diff'] = df_copy['RedWinsByDecision'] - df_copy['BlueWinsByDecision']
    df_copy['Loss_diff'] = df_copy['RedLosses'] - df_copy['BlueLosses']
    # Находим разницу между числом побед-поражений для обоих бойцов
    # df_copy['Red_Blue_wins_loss_diff'] = df_copy['Red_wins_loss_diff'] - df_copy['Blue_wins_loss_diff']
    # Находим разницу в статистических показателях для обоих бойцов
    df_copy['AvgSigStrLanded_diff'] = df_copy['RedAvgSigStrLanded'] - df_copy['BlueAvgSigStrLanded']
    df_copy['AvgSigStrPct_diff'] = df_copy['RedAvgSigStrPct'] - df_copy['BlueAvgSigStrPct']
    df_copy['TDAvgLanded_diff'] = df_copy['RedAvgTDLanded'] - df_copy['BlueAvgTDLanded']
    df_copy['AvgTDPct_diff'] = df_copy['RedAvgTDPct'] - df_copy['BlueAvgTDPct']
    df_copy['AvgSubAtt_diff'] = df_copy['RedAvgSubAtt'] - df_copy['BlueAvgSubAtt']
    # Находим разницу для других признаков
    df_copy['LongestWinStreak_diff'] = df_copy['RedLongestWinStreak'] - df_copy['BlueLongestWinStreak']
    df_copy['TotalRoundsFought_diff'] = df_copy['RedTotalRoundsFought'] - df_copy['BlueTotalRoundsFought']
    df_copy['TimeSinceLastFight_diff'] = df_copy['RedTimeSinceLastFight'] - df_copy['BlueTimeSinceLastFight']
    df_copy['Rank_diff'] = df_copy['RMatchWCRank'] - df_copy['BMatchWCRank']
    
    return df_copy

# Находим разницу в текущих сериях побед/поражений для двух бойцов
def red_blue_streak_diff_calc(row: pd.Series) -> pd.Series:
    """
    Вычисляет разность между текущими сериями побед/поражений для двух бойцов
    """
    # Если оба бойца одержали победы в предыдущих боях, то вычитаем серию побед бойца в синем из оной для бойца в красном
    if row['RedCurrentWinStreak'] > 0: 
        if row['BlueCurrentWinStreak'] > 0:
            return row['RedCurrentWinStreak'] - row['BlueCurrentWinStreak']
        else: # Если боец в синем идет после поражения, то добавляем количество его поражений к количетву побед бойца в красном
            return row['RedCurrentWinStreak'] + row['BlueCurrentLoseStreak']
    else: # Если боец в синем идет после победы, а оец в красном - после поражения, то вычитаем первое из второго
        if row['BlueCurrentWinStreak'] > 0:
            return -row['RedCurrentLoseStreak'] - row['BlueCurrentWinStreak']
        else:
            return -row['RedCurrentLoseStreak'] + row['BlueCurrentLoseStreak']     
        
# Убираем исходные признаки и оставляем только созданные
def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет ненужные признаки, оставляя только признаки-разности 
    """
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=[
                          'RedAge',
                          'RedHeightCms',
                          'RedReachCms',
                          'RedWeightLbs',
                          'BlueAge',
                          'BlueHeightCms',
                          'BlueReachCms',
                          'BlueWeightLbs',
                          'RedWins',
                          'RedLosses',
                          'BlueWins',
                          'BlueLosses',
                          'RedCurrentWinStreak',
                          'RedCurrentLoseStreak',
                          'RedTotalRoundsFought',
                          'BlueCurrentWinStreak',
                          'BlueCurrentLoseStreak',                                                               
                          'WeightClass',
                          'RedAvgSigStrLanded',
                          'RedAvgSigStrPct',
                          'BlueAvgSigStrLanded',
                          'BlueAvgSigStrPct',
                          'RedAvgTDLanded',
                          'RedAvgTDPct', 
                          'BlueAvgTDLanded',
                          'BlueAvgTDPct',
                          'RedAvgSubAtt',                          
                          'BlueAvgSubAtt',
                          'Gender',
                          'NumberOfRounds',
                          'RedWinsBySubmission',
                          'BlueWinsBySubmission',
                          'RedWinsByDecision',
                          'RedWinsByKO_TKO',
                          'BlueWinsByDecision',
                          'BlueWinsByKO_TKO',
                          'BlueTotalRoundsFought',
                          'RedLongestWinStreak',
                          'BlueLongestWinStreak',
                          'RedTimeSinceLastFight',
                          'BlueTimeSinceLastFight',
                          'RMatchWCRank',
                          'BMatchWCRank',
                          'RedFighter',
                          'BlueFighter',                         
                          ], axis=1)
    return df_copy


def run_xgboost_model_prediction(model: Any, df_test: np.array) -> Dict:
    """
    Запуск предсказания модели XGBoost. 
    Возвращает словарь:
    model: модель,
    y_pred: предсказанные метки класса,
    y_proba: вероятности предсказаний
    """
    # Получаем предсказания загруженной модели 
    y_pred = model.predict(df_test)
    y_proba = model.predict_proba(df_test)  

    return {'model': model,'y_pred': y_pred, 'y_proba': y_proba}



def run_rnn_model_prediction(model: Any, df_test: np.array) -> Dict:
    """
    Запуск предсказания модели RNN. 
    Возвращает словарь:
    model: модель,
    y_pred: предсказанные метки класса,
    y_proba: вероятности предсказаний
    """

    # Создаем DataLoader для тестовых данных
    test_dataloader = DataLoader(df_test, batch_size=32, shuffle=False)
    model.eval()# Переключаем модель в режим оценки
    y_pred = []
    y_proba = []
    with torch.no_grad():  # Отключаем обратное распространение ошибок
        for batch in test_dataloader:
            inputs = batch.float()
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            y_pred.extend(predicted.numpy().flatten())            
            y_proba.extend(outputs.numpy().flatten())     

   # Преобразовать списки в numpy массивы
    y_pred = np.array(y_pred)
    y_proba_1 = np.array(y_proba)
    y_proba_0 = 1 - np.array(y_proba)
    
    y_proba = np.stack((y_proba_0, y_proba_1), axis=1)  

    return {'model': model,'y_pred': y_pred, 'y_proba': y_proba}


def calc_odds(row: pd.Series) -> pd.Series:
    """
    Пересчет коэффициентов букмекеров
    из американского в европейский (десятичный) формат
    """
    for col in ['RedOdds', 'BlueOdds']:
        if row[col] > 0:
            row[col] = row[col] / 100 + 1
        else:
            row[col] = 100 / np.abs(row[col]) + 1
    return row

    
def calc_gain_expectation(row: pd.Series, bet_size: int) -> pd.Series:
    """
    Расчет математического ожидания выигрыша за каждую ставку
    """
    row['RedGainExpect'] = int(row['RedOdds'] * bet_size * row['ProbRedWins'])
    row['BlueGainExpect'] = int(row['BlueOdds'] * bet_size * row['ProbBlueWins'])

    return row


def show_predictions(df_test: pd.DataFrame, predictions: Dict, bet_size: int) -> pd.DataFrame:
    """
    Возвращает датафрейм с предсказаниями победителя боя и
    предсказанными вероятностями победы каждого из бойцов.
    Рассчитывает математическое ожидание выигрыша для обоих бойцов 
    """
    predictions_df = pd.DataFrame({
                                'RedFighter': df_test['RedFighter'],
                                'BlueFighter': df_test['BlueFighter'],
                                'RedOdds': df_test['RedOdds'],
                                'BlueOdds': df_test['BlueOdds'],
                                'ProbRedWins': np.round(predictions['y_proba'][:, 0], 2),
                                'ProbBlueWins': np.round(predictions['y_proba'][:, 1], 2)                                 
                                })    

    # Рассчитываем выигрыш для каждого боя
    predictions_df = predictions_df.apply(calc_odds, axis=1)   
    predictions_df = predictions_df.apply(calc_gain_expectation, bet_size=bet_size, axis=1)

    return predictions_df