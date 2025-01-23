import io
import copy
from fastapi import FastAPI, Depends, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Annotated
from models_store import get_models
from functions import create_new_features,\
                      red_blue_streak_diff_calc,\
                      drop_features,\
                      run_xgboost_model_prediction,\
                      run_rnn_model_prediction,\
                      show_predictions
                      
import joblib


router = APIRouter()

# Переменная для хранения загруженных данных
df_test: pd.DataFrame = pd.DataFrame()

# Модель, выбранная для предсказания
selected_model = None
selected_model_name = None


# Модель для данных в JSON
class FightData(BaseModel):
    RedFighter: str
    BlueFighter: str
    WeightClass: int
    Gender: int
    NumberOfRounds: float
    RedAge: int
    RedHeightCms: float
    RedReachCms: float
    RedWeightLbs: int
    RedStance: float
    RedWins: int
    RedWinsBySubmission: int
    RedCurrentWinStreak: int
    RedLosses: int
    RedCurrentLoseStreak: int
    RedAvgSigStrLanded: float
    RedAvgSigStrPct: float
    RedAvgSubAtt: float
    RedAvgTDLanded: float
    RedAvgTDPct: float
    RedTotalRoundsFought: int
    RedLongestWinStreak: int
    BlueAge: int
    BlueHeightCms: float
    BlueReachCms: float
    BlueWeightLbs: int
    BlueStance: float
    BlueWins: int
    BlueWinsBySubmission: int
    BlueCurrentWinStreak: int
    BlueLosses: int
    BlueCurrentLoseStreak: int
    BlueAvgSigStrLanded: float
    BlueAvgSigStrPct: float
    BlueAvgSubAtt: float
    BlueAvgTDLanded: float
    BlueAvgTDPct: float
    BlueTotalRoundsFought: int
    BlueLongestWinStreak: int
    RedOdds: float
    BlueOdds: float
    RMatchWCRank: float
    BMatchWCRank: float    
    RedWinsByDecision: int
    RedWinsByKO_TKO: int  
    BlueWinsByDecision: int 
    BlueWinsByKO_TKO: int  
    RedTimeSinceLastFight: int 
    BlueTimeSinceLastFight: int 



@router.post("/upload_fight_data_json/")
async def upload_json(data: List[FightData]):
    """
    Эндпоинт для загрузки данных о бое в формате JSON
    """
    global df_test

    try:
        # Преобразуем список JSON-объектов в DataFrame и сохраняем как оригинальные данные
        df_test = pd.DataFrame([item.model_dump() for item in data])

        return {"message": "JSON data loaded successfully", "dataframe": df_test.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing JSON data: {e}")


@router.post("/upload_fight_data_csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Эндпоинт для загрузки CSV-файла и преобразования его в DataFrame.
    """
    global df_test

    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Чтение содержимого CSV-файла и сохранение как оригинальные данные
        file_content = await file.read()
        df_test = pd.read_csv(io.StringIO(file_content.decode("utf-8")), index_col='Unnamed: 0')

        return {"message": "CSV file loaded successfully", "dataframe": df_test.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {e}")



@router.post("/select-model/{model_name}")
async def select_model(model_name: str, models: Dict[str, Any] = Depends(get_models)):
    global selected_model, selected_model_name

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found.")
    
    selected_model_name = model_name
    selected_model = models[model_name]  # Полностью копируем объект из словаря models
    return {"message": f"Model '{model_name}' selected successfully."}



# Эндпоинт для выполнения предсказания
@router.post("/predict")
async def predict():

    global df_test, selected_model
    # selected_model = models.get("selected_model")

    if df_test.empty:
        raise HTTPException(status_code=400, detail="No data loaded. Use /upload-json or /upload-csv to load data.")   

    if selected_model is None:
        raise HTTPException(status_code=400, detail="No model selected. Use /select-model to choose a model.")
    
    try:
        # Создаем копию данных, чтобы не изменять оригинальные данные
        test_data = copy.deepcopy(df_test)

        # Применяем функции, отвечающие за создание новых признаков и удаление ненужных
        test_data  = create_new_features(test_data)
        test_data ['Curr_streak_diff'] = test_data.apply(red_blue_streak_diff_calc, axis=1)
        test_data = drop_features(test_data)

        # Загружаем сохраненный скейлер и применяем его к тестовым данным
        scaler = joblib.load('models/ufc_stand_scaler.joblib')
        test_data = scaler.transform(test_data)

        # Получаем предсказания с выбранной моделью
        if selected_model_name == 'XGBoost':
            predictions = run_xgboost_model_prediction(selected_model, test_data)

        else:
            predictions = run_rnn_model_prediction(selected_model, test_data)

        # Выводим результаты предсказания
        predictions_df = show_predictions(df_test, predictions, 1000)

        # Рассчитываем сумму выигрыша
        # gain = calc_gain(predictions_df)
    
        return {
            "message": "Predictions successfully generated",
            "predictions": predictions_df.to_dict(orient="records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating predictions: {e}")
    

@router.get("/check-models/")
async def check_models(models: Annotated[Dict[str, Any], Depends(get_models)]):
    models = get_models()
    if not models:
        return {"message": "No models loaded"}
    return {"models": list(models.keys())}


# Регистрируем роутер
# app.include_router(router)
