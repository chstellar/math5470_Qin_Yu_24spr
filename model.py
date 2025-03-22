import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import os
import time
import warnings
import logging
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import datetime
warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    """Timer utility for tracking time taken by code blocks"""
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")

# File paths
DATA_PATH = '/scratch/users/jiamuyu/proj_5470/data'
OUTPUT_PATH = './output'
LOG_PATH = './logs'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# Set up logging
def setup_logging():
    """Set up logging to file and console"""
    log_file = f"{LOG_PATH}/home_credit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of a dataframe by converting data types"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

def safe_division(a, b, default=0):
    """Safely divide two values, handling zeros and NaNs"""
    try:
        result = a / b
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(default)
        return result
    except:
        if isinstance(a, (pd.Series, np.ndarray)) and isinstance(b, (pd.Series, np.ndarray)):
            # Element-wise division for Series or arrays
            mask = (b != 0)
            result = np.zeros_like(a, dtype=float)
            result[mask] = a[mask] / b[mask]
            return result
        else:
            # Scalar division
            return default if b == 0 else a / b

def save_features(feature_df, name, path):
    """Save a feature dataframe to disk"""
    feature_path = f"{path}/{name}.pkl"
    feature_df.to_pickle(feature_path)
    logger.info(f"Saved {name} features to disk ({feature_df.shape[0]} rows, {feature_df.shape[1]} columns)")

def load_features(name, path):
    """Load a feature dataframe from disk"""
    feature_path = f"{path}/{name}.pkl"
    if os.path.exists(feature_path):
        logger.info(f"Loading {name} features from disk")
        return pd.read_pickle(feature_path)
    return None

def load_data():
    """Load all datasets"""
    with timer("Loading datasets"):
        train_df = pd.read_csv(f'{DATA_PATH}/application_train.csv')
        test_df = pd.read_csv(f'{DATA_PATH}/application_test.csv')
        bureau = pd.read_csv(f'{DATA_PATH}/bureau.csv')
        bureau_balance = pd.read_csv(f'{DATA_PATH}/bureau_balance.csv')
        prev = pd.read_csv(f'{DATA_PATH}/previous_application.csv')
        pos = pd.read_csv(f'{DATA_PATH}/POS_CASH_balance.csv')
        installments = pd.read_csv(f'{DATA_PATH}/installments_payments.csv')
        cc_balance = pd.read_csv(f'{DATA_PATH}/credit_card_balance.csv')
        
    print(f'Train samples: {train_df.shape[0]}, Test samples: {test_df.shape[0]}')
    
    return train_df, test_df, bureau, bureau_balance, prev, pos, installments, cc_balance

def clean_feature_names(df):
    """Clean feature names to remove special characters that might cause problems in models"""
    cleaned_columns = {}
    for col in df.columns:
        # Replace special characters with underscores
        new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        # Ensure no duplicate column names after cleaning
        if new_col in cleaned_columns.values():
            new_col = new_col + '_' + str(sum(1 for x in cleaned_columns.values() if x.startswith(new_col)))
        cleaned_columns[col] = new_col
    
    # Rename columns
    df.columns = [cleaned_columns[col] for col in df.columns]
    return df, cleaned_columns

def process_application_data(train_df, test_df):
    """Process application data (train and test)"""
    with timer("Processing application data"):
        # Combine train and test for preprocessing
        df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
        print(f"Combined application data shape: {df.shape}")
        
        # Extract target and ids
        train_target = train_df['TARGET'] if 'TARGET' in train_df.columns else None
        train_id = train_df['SK_ID_CURR']
        test_id = test_df['SK_ID_CURR']
        
        # Remove TARGET from df if exists
        if 'TARGET' in df.columns:
            df.drop(columns=['TARGET'], inplace=True)
        
        # Feature engineering
        # Days employed anomaly (replace 365243 with NaN)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        
        # Credit income ratio - with safeguards
        df['CREDIT_INCOME_RATIO'] = safe_division(df['AMT_CREDIT'], df['AMT_INCOME_TOTAL'])
        df['ANNUITY_INCOME_RATIO'] = safe_division(df['AMT_ANNUITY'], df['AMT_INCOME_TOTAL'])
        df['CREDIT_TERM'] = safe_division(df['AMT_CREDIT'], df['AMT_ANNUITY'])
        df['DAYS_EMPLOYED_RATIO'] = safe_division(df['DAYS_EMPLOYED'], df['DAYS_BIRTH'])
        
        # Store original values of categorical fields before encoding
        if 'NAME_INCOME_TYPE' in df.columns:
            df['ORIG_NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].copy()
            # Flag if income source is working
            df['WORKING_INCOME_FLAG'] = np.where(df['NAME_INCOME_TYPE'] == 'Working', 1, 0)
        
        # Flag for own car age
        df['OWN_CAR_AGE_FLAG'] = np.where(df['OWN_CAR_AGE'].isnull(), 0, 1)
        
        # One-hot encoding for categorical features
        categorical_features = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_features, dummy_na=True, drop_first=True)
        
        # Fill missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(-999)
        
        # Split back into train and test
        train_df = df[df['SK_ID_CURR'].isin(train_id)].copy()
        test_df = df[df['SK_ID_CURR'].isin(test_id)].copy()
        
        del df
        gc.collect()
        
        print(f"Processed train shape: {train_df.shape}, test shape: {test_df.shape}")
        
        return train_df, test_df, train_target, train_id, test_id

def process_bureau_data(bureau, bureau_balance, train_id, test_id):
    """Process bureau data"""
    with timer("Processing bureau data"):
        # Process bureau_balance
        bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max', 'size', 'mean'],
            'STATUS': ['nunique', lambda x: x.value_counts().get('C', 0),
                       lambda x: x.value_counts().get('X', 0),
                       lambda x: x.value_counts().get('0', 0),
                       lambda x: x.value_counts().get('1', 0),
                       lambda x: x.value_counts().get('2', 0),
                       lambda x: x.value_counts().get('3', 0),
                       lambda x: x.value_counts().get('4', 0),
                       lambda x: x.value_counts().get('5', 0)]
        })
        bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
        del bb_agg
        gc.collect()
        
        # Process bureau
        # Save original CREDIT_ACTIVE column for later use
        bureau['CREDIT_ACTIVE_STATUS'] = bureau['CREDIT_ACTIVE'].copy()
        
        # Replace loan statuses with numbers
        status_mapping = {
            'C': 0,
            'X': 1,
            '0': 2,
            '1': 3,
            '2': 4,
            '3': 5,
            '4': 6,
            '5': 7
        }
        for col in bureau.columns:
            if 'STATUS' in col:
                bureau[col] = bureau[col].map(status_mapping)
        
        # One-hot encoding for categorical variables
        categorical_features = [col for col in bureau.columns if bureau[col].dtype == 'object']
        bureau = pd.get_dummies(bureau, columns=categorical_features, dummy_na=True, drop_first=True)
        
        # Calculate aggregation features
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['min', 'max', 'mean', 'std', 'count'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['min', 'mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['max', 'mean', 'sum'],
            'BB_MONTHS_BALANCE_MIN': ['min', 'max', 'mean'],
            'BB_MONTHS_BALANCE_MAX': ['min', 'max', 'mean'],
            'BB_MONTHS_BALANCE_SIZE': ['min', 'max', 'mean', 'sum'],
            'BB_MONTHS_BALANCE_MEAN': ['min', 'max', 'mean'],
            'BB_STATUS_NUNIQUE': ['min', 'max', 'mean'],
        })
        bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        
        # Find the correct column names after one-hot encoding
        active_col = [col for col in bureau.columns if 'CREDIT_ACTIVE' in col and 'Active' in col]
        closed_col = [col for col in bureau.columns if 'CREDIT_ACTIVE' in col and 'Closed' in col]
        
        # Active and closed loans - use the first matching column if found
        if active_col:
            active = bureau[bureau[active_col[0]] == 1]
            active_agg = active.groupby('SK_ID_CURR').agg({
                'DAYS_CREDIT': ['min', 'max', 'mean', 'count'],
                'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            })
            active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
            bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        else:
            # Alternative approach: use CREDIT_ACTIVE_STATUS
            active = bureau[bureau['CREDIT_ACTIVE_STATUS'] == 'Active']
            if len(active) > 0:
                active_agg = active.groupby('SK_ID_CURR').agg({
                    'DAYS_CREDIT': ['min', 'max', 'mean', 'count'],
                    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                })
                active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
                bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        
        if closed_col:
            closed = bureau[bureau[closed_col[0]] == 1]
            closed_agg = closed.groupby('SK_ID_CURR').agg({
                'DAYS_CREDIT': ['min', 'max', 'mean', 'count'],
                'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            })
            closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
            bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        else:
            # Alternative approach: use CREDIT_ACTIVE_STATUS
            closed = bureau[bureau['CREDIT_ACTIVE_STATUS'] == 'Closed']
            if len(closed) > 0:
                closed_agg = closed.groupby('SK_ID_CURR').agg({
                    'DAYS_CREDIT': ['min', 'max', 'mean', 'count'],
                    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                })
                closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
                bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        
        # Fill NaN values
        bureau_agg = bureau_agg.fillna(0)
        
        # Create combined dataset with all IDs
        all_ids = pd.DataFrame({'SK_ID_CURR': pd.concat([train_id, test_id], axis=0).unique()})
        bureau_features = all_ids.merge(bureau_agg, on='SK_ID_CURR', how='left')
        bureau_features.fillna(0, inplace=True)
        
        print(f"Bureau features shape: {bureau_features.shape}")
        
        return bureau_features

def process_previous_applications(prev, train_id, test_id):
    """Process previous applications data"""
    with timer("Processing previous applications"):
        # Store original contract status BEFORE one-hot encoding
        if 'NAME_CONTRACT_STATUS' in prev.columns:
            prev['ORIG_CONTRACT_STATUS'] = prev['NAME_CONTRACT_STATUS'].copy()
        else:
            logger.warning("Column 'NAME_CONTRACT_STATUS' not found in previous_application.csv")
        
        # One-hot encoding for categorical variables
        categorical_features = [col for col in prev.columns if prev[col].dtype == 'object']
        prev = pd.get_dummies(prev, columns=categorical_features, dummy_na=True, drop_first=True)
        
        # Add features
        prev['APP_CREDIT_RATIO'] = safe_division(prev['AMT_APPLICATION'], prev['AMT_CREDIT'])
        prev['CREDIT_TO_GOODS_RATIO'] = safe_division(prev['AMT_CREDIT'], prev['AMT_GOODS_PRICE'])
        
        # Aggregate features
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': ['nunique', 'count'],
            'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
            'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
            'APP_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
            'CREDIT_TO_GOODS_RATIO': ['min', 'max', 'mean', 'var'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        })
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        
        # Conditionally process approved/refused applications based on available columns
        if 'ORIG_CONTRACT_STATUS' in prev.columns:
            # Process using original stored values
            approved = prev[prev['ORIG_CONTRACT_STATUS'] == 'Approved']
            if len(approved) > 0:
                approved_agg = approved.groupby('SK_ID_CURR').agg({
                    'SK_ID_PREV': ['nunique', 'count'],
                    'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
                    'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
                })
                approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
                prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
            
            refused = prev[prev['ORIG_CONTRACT_STATUS'] == 'Refused']
            if len(refused) > 0:
                refused_agg = refused.groupby('SK_ID_CURR').agg({
                    'SK_ID_PREV': ['nunique', 'count'],
                    'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
                    'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
                })
                refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
                prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        else:
            # Find the correct column names after one-hot encoding for status
            approved_col = [col for col in prev.columns if 'NAME_CONTRACT_STATUS' in col and 'Approved' in col]
            refused_col = [col for col in prev.columns if 'NAME_CONTRACT_STATUS' in col and 'Refused' in col]
            
            # Process if columns are found
            if approved_col:
                approved = prev[prev[approved_col[0]] == 1]
                approved_agg = approved.groupby('SK_ID_CURR').agg({
                    'SK_ID_PREV': ['nunique', 'count'],
                    'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
                    'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
                })
                approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
                prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
            
            if refused_col:
                refused = prev[prev[refused_col[0]] == 1]
                refused_agg = refused.groupby('SK_ID_CURR').agg({
                    'SK_ID_PREV': ['nunique', 'count'],
                    'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
                    'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
                })
                refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
                prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        
        # Fill NaN values
        prev_agg = prev_agg.fillna(0)
        
        # Create combined dataset with all IDs
        all_ids = pd.DataFrame({'SK_ID_CURR': pd.concat([train_id, test_id], axis=0).unique()})
        prev_features = all_ids.merge(prev_agg, on='SK_ID_CURR', how='left')
        prev_features.fillna(0, inplace=True)
        
        print(f"Previous applications features shape: {prev_features.shape}")
        
        return prev_features

def process_pos_cash(pos, train_id, test_id):
    """Process POS_CASH_balance data"""
    with timer("Processing POS_CASH_balance"):
        # Store original contract status
        if 'NAME_CONTRACT_STATUS' in pos.columns:
            pos['ORIG_CONTRACT_STATUS'] = pos['NAME_CONTRACT_STATUS'].copy()
            
        # One-hot encoding for categorical variables
        categorical_features = [col for col in pos.columns if pos[col].dtype == 'object']
        pos = pd.get_dummies(pos, columns=categorical_features, dummy_na=True, drop_first=True)
        
        # Aggregate features
        pos_agg = pos.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': ['nunique', 'count'],
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
            'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum']
        })
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        
        # Latest POS per loan - safer approach that doesn't depend on specific column names
        pos_sorted = pos.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
        pos_latest = pos_sorted.groupby('SK_ID_PREV').last().reset_index()
        
        # Aggregate for latest POS records
        pos_latest_agg = pos_latest.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        })
        pos_latest_agg.columns = pd.Index(['POS_LATEST_' + e[0] + "_" + e[1].upper() for e in pos_latest_agg.columns.tolist()])
        
        # Combine features
        pos_features = pos_agg.join(pos_latest_agg, how='left', on='SK_ID_CURR')
        
        # Fill NaN values
        pos_features = pos_features.fillna(0)
        
        # Create combined dataset with all IDs
        all_ids = pd.DataFrame({'SK_ID_CURR': pd.concat([train_id, test_id], axis=0).unique()})
        pos_features = all_ids.merge(pos_features, on='SK_ID_CURR', how='left')
        pos_features.fillna(0, inplace=True)
        
        print(f"POS cash features shape: {pos_features.shape}")
        
        return pos_features

def process_credit_card(cc_balance, train_id, test_id):
    """Process credit_card_balance data"""
    with timer("Processing credit_card_balance"):
        # Store original contract status
        if 'NAME_CONTRACT_STATUS' in cc_balance.columns:
            cc_balance['ORIG_CONTRACT_STATUS'] = cc_balance['NAME_CONTRACT_STATUS'].copy()
            
        # One-hot encoding for categorical variables
        categorical_features = [col for col in cc_balance.columns if cc_balance[col].dtype == 'object']
        cc_balance = pd.get_dummies(cc_balance, columns=categorical_features, dummy_na=True, drop_first=True)
        
        # Feature engineering - with safeguards
        cc_balance['CREDIT_PAYMENT_RATIO'] = safe_division(cc_balance['AMT_PAYMENT_CURRENT'], cc_balance['AMT_PAYMENT_TOTAL_CURRENT'])
        cc_balance['CREDIT_DRAWINGS_RATIO'] = safe_division(cc_balance['AMT_DRAWINGS_CURRENT'], cc_balance['AMT_CREDIT_LIMIT_ACTUAL'])
        
        # Latest credit card record per loan
        cc_sorted = cc_balance.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
        cc_latest = cc_sorted.groupby('SK_ID_PREV').last().reset_index()
        
        # Aggregate features
        cc_agg = cc_balance.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': ['nunique', 'count'],
            'MONTHS_BALANCE': ['min', 'max', 'size', 'mean'],
            'AMT_BALANCE': ['min', 'max', 'mean', 'sum', 'var'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
            'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'CREDIT_PAYMENT_RATIO': ['min', 'max', 'mean', 'var'],
            'CREDIT_DRAWINGS_RATIO': ['min', 'max', 'mean', 'var']
        })
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        
        # Aggregate for latest credit card records
        cc_latest_agg = cc_latest.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        })
        cc_latest_agg.columns = pd.Index(['CC_LATEST_' + e[0] + "_" + e[1].upper() for e in cc_latest_agg.columns.tolist()])
        
        # Combine features
        cc_features = cc_agg.join(cc_latest_agg, how='left', on='SK_ID_CURR')
        
        # Fill NaN values
        cc_features = cc_features.fillna(0)
        
        # Create combined dataset with all IDs
        all_ids = pd.DataFrame({'SK_ID_CURR': pd.concat([train_id, test_id], axis=0).unique()})
        cc_features = all_ids.merge(cc_features, on='SK_ID_CURR', how='left')
        cc_features.fillna(0, inplace=True)
        
        print(f"Credit card features shape: {cc_features.shape}")
        
        return cc_features
        
def process_installments(installments, train_id, test_id):
    """Process installments_payments data"""
    with timer("Processing installments_payments"):
        # Feature engineering
        installments['PAYMENT_RATIO'] = safe_division(installments['AMT_PAYMENT'], installments['AMT_INSTALMENT'])
        installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
        installments['DAYS_LATE'] = np.maximum(installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT'], 0)
        installments['DAYS_EARLY'] = np.maximum(installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT'], 0)
        
        # Aggregate features
        inst_agg = installments.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': ['nunique', 'count'],
            'NUM_INSTALMENT_VERSION': ['nunique', 'max'],
            'DAYS_INSTALMENT': ['min', 'max', 'mean'],
            'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
            'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'PAYMENT_RATIO': ['min', 'max', 'mean', 'var'],
            'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
            'DAYS_LATE': ['max', 'mean', 'sum'],
            'DAYS_EARLY': ['max', 'mean', 'sum']
        })
        inst_agg.columns = pd.Index(['INST_' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
        
        # Calculate late payment features
        late_payments = installments[installments['DAYS_LATE'] > 0]
        if len(late_payments) > 0:
            late_agg = late_payments.groupby('SK_ID_CURR').agg({
                'SK_ID_PREV': ['nunique', 'count'],
                'DAYS_LATE': ['max', 'mean', 'sum'],
                'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
            })
            late_agg.columns = pd.Index(['LATE_' + e[0] + "_" + e[1].upper() for e in late_agg.columns.tolist()])
            inst_agg = inst_agg.join(late_agg, how='left', on='SK_ID_CURR')
        
        # Fill NaN values
        inst_agg = inst_agg.fillna(0)
        
        # Create combined dataset with all IDs
        all_ids = pd.DataFrame({'SK_ID_CURR': pd.concat([train_id, test_id], axis=0).unique()})
        inst_features = all_ids.merge(inst_agg, on='SK_ID_CURR', how='left')
        inst_features.fillna(0, inplace=True)
        
        print(f"Installments features shape: {inst_features.shape}")
        
        return inst_features

def combine_features(train_df, test_df, bureau_features, prev_features, pos_features, inst_features, cc_features,
                     train_target, train_id, test_id):
    """Combine all feature sets"""
    with timer("Combining features"):
        # Combine all features for training set
        train_features = train_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        train_features = train_features.merge(prev_features, on='SK_ID_CURR', how='left')
        train_features = train_features.merge(pos_features, on='SK_ID_CURR', how='left')
        train_features = train_features.merge(inst_features, on='SK_ID_CURR', how='left')
        train_features = train_features.merge(cc_features, on='SK_ID_CURR', how='left')
        
        # Combine all features for testing set
        test_features = test_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        test_features = test_features.merge(prev_features, on='SK_ID_CURR', how='left')
        test_features = test_features.merge(pos_features, on='SK_ID_CURR', how='left')
        test_features = test_features.merge(inst_features, on='SK_ID_CURR', how='left')
        test_features = test_features.merge(cc_features, on='SK_ID_CURR', how='left')
        
        # Fill remaining NaN values
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)
        
        print(f"Final training set shape: {train_features.shape}")
        print(f"Final testing set shape: {test_features.shape}")
        
        # Extract feature names
        feat_cols = [col for col in train_features.columns if col != 'SK_ID_CURR']
        
        return train_features, test_features, feat_cols, train_target, train_id, test_id

def safe_division(a, b, default=0):
    """Safely divide two values, handling zeros and NaNs"""
    try:
        result = a / b
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(default)
        return result
    except:
        if isinstance(a, (pd.Series, np.ndarray)) and isinstance(b, (pd.Series, np.ndarray)):
            # Element-wise division for Series or arrays
            mask = (b != 0)
            result = np.zeros_like(a, dtype=float)
            result[mask] = a[mask] / b[mask]
            return result
        else:
            # Scalar division
            return default if b == 0 else a / b

def optimize_dataframe(df):
    """Optimize dataframe memory usage"""
    result = reduce_mem_usage(df.copy())
    del df
    gc.collect()
    return result

def save_features(feature_df, name, path):
    """Save a feature dataframe to disk"""
    feature_df.to_pickle(f"{path}/{name}.pkl")
    print(f"Saved {name} features to disk")

def load_features(name, path):
    """Load a feature dataframe from disk"""
    if os.path.exists(f"{path}/{name}.pkl"):
        print(f"Loading {name} features from disk")
        return pd.read_pickle(f"{path}/{name}.pkl")
    return None

def select_features(train_data, target, feat_cols, n_features=200):
    """Select most important features using a simple model"""
    # Train a simple model
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=32,
        colsample_bytree=0.8,
        subsample=0.8,
        max_depth=7,
        random_state=42
    )
    model.fit(train_data[feat_cols], target)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = importance.head(n_features)['feature'].tolist()
    
    print(f"Selected {len(selected_features)} features out of {len(feat_cols)}")
    return selected_features

def get_folds(data, target, n_folds=5, stratify=True, seed=42):
    """Create fold indices for cross-validation"""
    if stratify:
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(data, target))
    else:
        return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(data))

def generate_submission_file(test_id, test_preds, model_name="model"):
    """Generate a submission file in the required format"""
    submission = pd.DataFrame({"SK_ID_CURR": test_id, "TARGET": test_preds})
    
    # Make sure SK_ID_CURR is an integer (sometimes it can be converted to float)
    submission["SK_ID_CURR"] = submission["SK_ID_CURR"].astype(int)
    
    # Save the submission
    submission_file = f"{OUTPUT_PATH}/{model_name}_submission_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(submission_file, index=False)
    
    print(f"Submission file saved to {submission_file}")
    return submission

def analyze_model_performance(models, train_features, test_features, train_target, test_id, output_path):
    """Analyze model performance across different segments"""
    results = {}
    
    # Create segments based on various features
    segments = {}
    
    # By income level
    if 'AMT_INCOME_TOTAL' in train_features.columns:
        train_features['INCOME_GROUP'] = pd.qcut(train_features['AMT_INCOME_TOTAL'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        segments['INCOME_GROUP'] = train_features['INCOME_GROUP']
    
    # By age
    if 'DAYS_BIRTH' in train_features.columns:
        age_years = -train_features['DAYS_BIRTH'] / 365
        train_features['AGE_GROUP'] = pd.cut(age_years, [20, 35, 50, 65, 100], labels=['20-35', '35-50', '50-65', '65+'])
        segments['AGE_GROUP'] = train_features['AGE_GROUP']
    
    # By employment length
    if 'DAYS_EMPLOYED' in train_features.columns:
        emp_years = -train_features['DAYS_EMPLOYED'] / 365
        train_features['EMPLOYMENT_GROUP'] = pd.cut(emp_years, [-1, 1, 5, 10, 100], labels=['<1 year', '1-5 years', '5-10 years', '10+ years'])
        segments['EMPLOYMENT_GROUP'] = train_features['EMPLOYMENT_GROUP']
    
    # By credit amount
    if 'AMT_CREDIT' in train_features.columns:
        train_features['CREDIT_GROUP'] = pd.qcut(train_features['AMT_CREDIT'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        segments['CREDIT_GROUP'] = train_features['CREDIT_GROUP']
    
    # Calculate metrics for each segment
    for segment_name, segment_values in segments.items():
        results[segment_name] = {}
        
        for segment_value in segment_values.unique():
            if pd.isna(segment_value):
                continue
                
            segment_mask = segment_values == segment_value
            segment_target = train_target[segment_mask]
            
            if len(segment_target) < 10:  # Skip very small segments
                continue
                
            results[segment_name][segment_value] = {
                'count': len(segment_target),
                'target_rate': segment_target.mean()
            }
            
            # Calculate model performance on segment
            for model_name, model_preds in models.items():
                segment_preds = model_preds[segment_mask]
                if len(segment_preds) > 0:
                    results[segment_name][segment_value][f'{model_name}_auc'] = roc_auc_score(segment_target, segment_preds)
    
    # Save results to CSV for further analysis
    segment_results = []
    for segment_name, segment_data in results.items():
        for segment_value, metrics in segment_data.items():
            row = {'Segment': segment_name, 'Value': segment_value}
            row.update(metrics)
            segment_results.append(row)
            
    segment_df = pd.DataFrame(segment_results)
    segment_df.to_csv(f"{output_path}/segment_performance.csv", index=False)
    
    # Create visualizations
    for segment_name in results.keys():
        plt.figure(figsize=(12, 6))
        
        # Filter data for plotting
        plot_data = segment_df[segment_df['Segment'] == segment_name]
        
        # Create bar chart of segment target rates
        ax1 = plt.subplot(1, 2, 1)
        sns.barplot(x='Value', y='target_rate', data=plot_data, ax=ax1)
        ax1.set_title(f'Default Rate by {segment_name}')
        ax1.set_ylabel('Default Rate')
        ax1.set_xlabel(segment_name)
        
        # Create bar chart of model performance by segment
        ax2 = plt.subplot(1, 2, 2)
        model_cols = [col for col in plot_data.columns if 'auc' in col]
        if model_cols:
            plot_data_melted = pd.melt(
                plot_data, 
                id_vars=['Value'], 
                value_vars=model_cols,
                var_name='Model', 
                value_name='AUC'
            )
            plot_data_melted['Model'] = plot_data_melted['Model'].str.replace('_auc', '')
            sns.barplot(x='Value', y='AUC', hue='Model', data=plot_data_melted, ax=ax2)
            ax2.set_title(f'Model Performance by {segment_name}')
            ax2.set_ylabel('AUC')
            ax2.set_xlabel(segment_name)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/segment_{segment_name}.png")
        plt.close()
        
    return segment_df

def compare_model_predictions(models, train_features, train_target, test_features, test_id, output_path):
    """Compare predictions from different models"""
    if len(models) < 2:
        print("Need at least 2 models to compare predictions")
        return
    
    # Create correlation matrix of model predictions
    model_names = list(models.keys())
    corr_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            corr_matrix[i, j] = np.corrcoef(models[model1], models[model2])[0, 1]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, cmap='YlGnBu')
    plt.title('Correlation of Model Predictions')
    plt.tight_layout()
    plt.savefig(f"{output_path}/model_correlation.png")
    plt.close()
    
    # Analyze where models disagree
    disagreement_df = pd.DataFrame()
    
    for model_name, model_preds in models.items():
        disagreement_df[model_name] = model_preds
        
    disagreement_df['target'] = train_target.values
    disagreement_df['pred_variance'] = disagreement_df[model_names].var(axis=1)
    
    # Sort by prediction variance (highest disagreement first)
    disagreement_df = disagreement_df.sort_values('pred_variance', ascending=False)
    
    # Save top disagreement cases
    disagreement_df.head(1000).to_csv(f"{output_path}/model_disagreements.csv", index=False)
    
    # Plot histogram of prediction variances
    plt.figure(figsize=(10, 6))
    sns.histplot(disagreement_df['pred_variance'], kde=True)
    plt.title('Distribution of Model Prediction Variance')
    plt.xlabel('Variance between model predictions')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_path}/prediction_variance.png")
    plt.close()
    
    return disagreement_df

def plot_roc_curves(models, train_target, output_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, model_preds in models.items():
        fpr, tpr, _ = roc_curve(train_target, model_preds)
        auc = roc_auc_score(train_target, model_preds)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    
    # Add random classifier reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/roc_curves.png")
    plt.close()

def analyze_feature_correlations(train_features, train_target, n_features=30, output_path='.'):
    """Analyze correlations between features and with target"""
    # Select top features correlated with target
    feature_correlations = pd.DataFrame(index=train_features.columns)
    feature_correlations['target_correlation'] = [
        np.corrcoef(train_features[col].fillna(0), train_target)[0, 1] for col in train_features.columns
    ]
    
    top_features = feature_correlations.abs().sort_values('target_correlation', ascending=False).head(n_features).index
    
    # Create correlation matrix for top features
    corr_matrix = train_features[top_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='YlGnBu', annot=False, center=0, square=True)
    plt.title(f'Correlation Matrix of Top {n_features} Features')
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_correlation.png")
    plt.close()
    
    # Save feature-target correlations
    feature_correlations.sort_values('target_correlation', ascending=False).to_csv(
        f"{output_path}/feature_target_correlation.csv"
    )
    
    return feature_correlations

def analyze_feature_importance(feature_importance_df, train_features, train_target, output_path):
    """Analyze feature importance patterns"""
    if feature_importance_df.empty:
        print("No feature importance data available")
        return
        
    # Calculate mean importance by feature
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
    mean_importance = mean_importance.sort_values('importance', ascending=False)
    
    # Get top 20 features
    top_features = mean_importance.head(20)['feature'].tolist()
    
    # Create summary dataframe of top features
    top_features_df = pd.DataFrame()
    top_features_df['feature'] = top_features
    
    # Add importance
    for feature in top_features:
        importance = mean_importance[mean_importance['feature'] == feature]['importance'].values[0]
        top_features_df.loc[top_features_df['feature'] == feature, 'importance'] = importance
    
    # Add correlation with target
    for feature in top_features:
        if feature in train_features.columns:
            correlation = np.corrcoef(train_features[feature].fillna(0), train_target)[0, 1]
            top_features_df.loc[top_features_df['feature'] == feature, 'target_correlation'] = correlation
    
    # Calculate stability of importance across folds
    fold_stability = feature_importance_df.groupby('feature')['importance'].std() / feature_importance_df.groupby('feature')['importance'].mean()
    fold_stability = fold_stability.reset_index()
    fold_stability.columns = ['feature', 'importance_stability']
    
    # Add stability metrics
    for feature in top_features:
        stability = fold_stability[fold_stability['feature'] == feature]['importance_stability'].values
        if len(stability) > 0:
            top_features_df.loc[top_features_df['feature'] == feature, 'stability_score'] = stability[0]
    
    # Save detailed feature analysis
    top_features_df.to_csv(f"{output_path}/top_features_analysis.csv", index=False)
    
    # Plot top feature distributions
    for feature in top_features[:10]:  # Plot top 10 only to avoid too many plots
        if feature in train_features.columns:
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Distribution by target value
            plt.subplot(1, 2, 1)
            sns.histplot(
                data=pd.DataFrame({
                    'feature': train_features[feature],
                    'target': train_target
                }), 
                x='feature', 
                hue='target',
                kde=True,
                stat='density'
            )
            plt.title(f'Distribution of {feature} by Target')
            
            # Plot 2: Box plot by target value
            plt.subplot(1, 2, 2)
            sns.boxplot(
                x='target', 
                y='feature',
                data=pd.DataFrame({
                    'feature': train_features[feature],
                    'target': train_target
                })
            )
            plt.title(f'Box Plot of {feature} by Target')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/feature_distribution_{feature}.png")
            plt.close()
    
    return top_features_df

def get_model_params(model_type='lgbm', fast_mode=False):
    """Get model parameters based on model type and mode"""
    if fast_mode:
        if model_type == 'lgbm':
            return {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 32,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'max_depth': 7,
                'reg_alpha': 0.04,
                'reg_lambda': 0.07,
                'min_split_gain': 0.02,
                'min_child_weight': 40,
                'random_state': 42
            }
        elif model_type == 'xgb':
            # Simplified parameters for XGBoost
            return {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        else:  # catboost
            return {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42
            }
    else:
        if model_type == 'lgbm':
            return {
                'n_estimators': 10000,
                'learning_rate': 0.02,
                'num_leaves': 34,
                'colsample_bytree': 0.9497036,
                'subsample': 0.8715623,
                'max_depth': 8,
                'reg_alpha': 0.041545473,
                'reg_lambda': 0.0735294,
                'min_split_gain': 0.0222415,
                'min_child_weight': 39.3259775,
                'random_state': 42
            }
        elif model_type == 'xgb':
            # Simplified parameters for XGBoost
            return {
                'max_depth': 6,
                'learning_rate': 0.02,
                'n_estimators': 10000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        else:  # catboost
            return {
                'iterations': 10000,
                'learning_rate': 0.02,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42
            }

def setup_logging(log_path):
    """Set up logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_model_safely(model_type, X_train, y_train, X_valid, y_valid, params, fold):
    """Train a model with proper error handling"""
    try:
        if model_type == 'lgbm':
            model = lgb.LGBMClassifier(**params)
            try:
                # Try with callbacks (newer LightGBM versions)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(stopping_rounds=200)]
                )
            except:
                # Fall back to early_stopping_rounds (older LightGBM versions)
                print(f"Warning: Using fallback method for LightGBM in fold {fold}")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',
                    early_stopping_rounds=200,
                    verbose=100
                )
        elif model_type == 'xgb':
            # Most basic fit for XGBoost - highest compatibility
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
        else:  # catboost
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                use_best_model=True,
                verbose=200
            )
        return model
    except Exception as e:
        print(f"Error training {model_type} in fold {fold}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def model_training_lgbm(train_df, test_df, features, target, train_id, test_id, folds=5, fast_mode=False):
    """Train a LightGBM model and generate predictions"""
    with timer("LightGBM training"):
        # Prepare datasets
        train_data = train_df[features]
        test_data = test_df[features]
        
        # Clean feature names
        train_data, cleaned_columns = clean_feature_names(train_data)
        # Apply same cleaning to test data columns
        test_data.columns = [cleaned_columns[col] if col in cleaned_columns else col for col in test_data.columns]
        
        # Get model parameters
        params = get_model_params('lgbm', fast_mode)
        
        # Setup folds
        fold_indices = get_folds(train_data, target, n_folds=folds)
        
        # Initialize
        oof_preds = np.zeros(train_data.shape[0])
        test_preds = np.zeros(test_data.shape[0])
        feature_importance_df = pd.DataFrame()
        
        # Train across folds
        for fold_, (trn_idx, val_idx) in enumerate(fold_indices):
            print(f'LGBM Fold {fold_ + 1}')
            
            X_train, y_train = train_data.iloc[trn_idx], target.iloc[trn_idx]
            X_valid, y_valid = train_data.iloc[val_idx], target.iloc[val_idx]
            
            # Train the model
            model = train_model_safely('lgbm', X_train, y_train, X_valid, y_valid, params, fold_)
            
            if model is None:
                print(f"Skipping fold {fold_ + 1} due to training error")
                continue
                
            # Make predictions
            oof_preds[val_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds += model.predict_proba(test_data)[:, 1] / folds
            
            # Save feature importance
            try:
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = X_train.columns.tolist()
                fold_importance_df["importance"] = model.feature_importances_
                fold_importance_df["fold"] = fold_ + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            except:
                print("Could not calculate feature importance for this fold")
            
            # Calculate AUC for this fold
            print(f'Fold {fold_ + 1} AUC: {roc_auc_score(y_valid, oof_preds[val_idx])}')
            
            del X_train, X_valid, y_train, y_valid, model
            gc.collect()
        
        # Calculate overall AUC
        print(f'Full OOF AUC: {roc_auc_score(target, oof_preds)}')
        
        # Create submission
        submission = generate_submission_file(test_id, test_preds, "lgbm")
        
        # Feature importance - only if we have data
        if not feature_importance_df.empty and 'feature' in feature_importance_df.columns:
            display_importances(feature_importance_df)
        
        return submission, feature_importance_df, oof_preds

def model_training_xgb(train_df, test_df, features, target, train_id, test_id, folds=5, fast_mode=False):
    """Train an XGBoost model and generate predictions"""
    with timer("XGBoost training"):
        # Prepare datasets
        train_data = train_df[features]
        test_data = test_df[features]
        
        # Clean feature names
        train_data, cleaned_columns = clean_feature_names(train_data)
        # Apply same cleaning to test data columns
        test_data.columns = [cleaned_columns[col] if col in cleaned_columns else col for col in test_data.columns]
        
        # Get model parameters
        params = get_model_params('xgb', fast_mode)
        
        # Setup folds
        fold_indices = get_folds(train_data, target, n_folds=folds)
        
        # Initialize
        oof_preds = np.zeros(train_data.shape[0])
        test_preds = np.zeros(test_data.shape[0])
        feature_importance_df = pd.DataFrame()
        
        # Train across folds
        for fold_, (trn_idx, val_idx) in enumerate(fold_indices):
            print(f'XGB Fold {fold_ + 1}')
            
            X_train, y_train = train_data.iloc[trn_idx], target.iloc[trn_idx]
            X_valid, y_valid = train_data.iloc[val_idx], target.iloc[val_idx]
            
            # Train the model
            model = train_model_safely('xgb', X_train, y_train, X_valid, y_valid, params, fold_)
            
            if model is None:
                print(f"Skipping fold {fold_ + 1} due to training error")
                continue
                
            # Make predictions
            oof_preds[val_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds += model.predict_proba(test_data)[:, 1] / folds
            
            # Save feature importance
            try:
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = X_train.columns.tolist()
                fold_importance_df["importance"] = model.feature_importances_
                fold_importance_df["fold"] = fold_ + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            except:
                print("Could not calculate feature importance for this fold")
            
            # Calculate AUC for this fold
            print(f'Fold {fold_ + 1} AUC: {roc_auc_score(y_valid, oof_preds[val_idx])}')
            
            del X_train, X_valid, y_train, y_valid, model
            gc.collect()
        
        # Calculate overall AUC
        print(f'Full OOF AUC: {roc_auc_score(target, oof_preds)}')
        
        # Create submission
        submission = generate_submission_file(test_id, test_preds, "xgb")
        
        # Feature importance - only if we have data
        if not feature_importance_df.empty and 'feature' in feature_importance_df.columns:
            display_importances(feature_importance_df)
        
        return submission, feature_importance_df, oof_preds

def model_training_catboost(train_df, test_df, features, target, train_id, test_id, folds=5, fast_mode=False):
    """Train a CatBoost model and generate predictions"""
    with timer("CatBoost training"):
        # Prepare datasets
        train_data = train_df[features]
        test_data = test_df[features]
        
        # Clean feature names
        train_data, cleaned_columns = clean_feature_names(train_data)
        # Apply same cleaning to test data columns
        test_data.columns = [cleaned_columns[col] if col in cleaned_columns else col for col in test_data.columns]
        
        # Get model parameters
        params = get_model_params('catboost', fast_mode)
        
        # Setup folds
        fold_indices = get_folds(train_data, target, n_folds=folds)
        
        # Initialize
        oof_preds = np.zeros(train_data.shape[0])
        test_preds = np.zeros(test_data.shape[0])
        feature_importance_df = pd.DataFrame()
        
        # Train across folds
        for fold_, (trn_idx, val_idx) in enumerate(fold_indices):
            print(f'CatBoost Fold {fold_ + 1}')
            
            X_train, y_train = train_data.iloc[trn_idx], target.iloc[trn_idx]
            X_valid, y_valid = train_data.iloc[val_idx], target.iloc[val_idx]
            
            # Train the model
            model = train_model_safely('catboost', X_train, y_train, X_valid, y_valid, params, fold_)
            
            if model is None:
                print(f"Skipping fold {fold_ + 1} due to training error")
                continue
                
            # Make predictions
            oof_preds[val_idx] = model.predict_proba(X_valid)[:, 1]
            test_preds += model.predict_proba(test_data)[:, 1] / folds
            
            # Save feature importance
            try:
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = X_train.columns.tolist()
                fold_importance_df["importance"] = model.feature_importances_
                fold_importance_df["fold"] = fold_ + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            except:
                print("Could not calculate feature importance for this fold")
            
            # Calculate AUC for this fold
            print(f'Fold {fold_ + 1} AUC: {roc_auc_score(y_valid, oof_preds[val_idx])}')
            
            del X_train, X_valid, y_train, y_valid, model
            gc.collect()
        
        # Calculate overall AUC
        print(f'Full OOF AUC: {roc_auc_score(target, oof_preds)}')
        
        # Create submission
        submission = generate_submission_file(test_id, test_preds, "catboost")
        
        # Feature importance - only if we have data
        if not feature_importance_df.empty and 'feature' in feature_importance_df.columns:
            display_importances(feature_importance_df)
        
        return submission, feature_importance_df, oof_preds

def blend_models(submissions, test_id):
    """Blend multiple models' predictions"""
    with timer("Blending models"):
        # Equal weighting for all models
        blend_pred = np.zeros(len(submissions[0]))
        
        for submission in submissions:
            blend_pred += submission / len(submissions)
        
        # Create submission
        blend_submission = generate_submission_file(test_id, blend_pred, "blended")
        
        return blend_submission

def display_importances(feature_importance_df):
    """Display feature importances"""
    try:
        # Check if we have the required columns
        if 'feature' not in feature_importance_df.columns or 'importance' not in feature_importance_df.columns:
            print("Feature importance dataframe does not have required columns")
            return
            
        # Sort features by importance
        mean_gain = feature_importance_df[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)
        
        # Plot top 50 features (or fewer if we have less)
        n_features = min(50, len(mean_gain))
        plt.figure(figsize=(10, max(5, n_features/2)))
        sns.barplot(x='importance', y='feature', 
                   data=feature_importance_df.sort_values(by='importance', ascending=False).head(n_features))
        plt.title(f'Features (Top {n_features} by importance)')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PATH}/feature_importance.png')
    except Exception as e:
        print(f"Error generating feature importance plot: {str(e)}")

def main(fast_mode=False, feature_selection=False, n_features=200, analysis_mode=False):
    """Main execution function with options for faster runs and analysis"""
    print(f"Home Credit Default Risk Solution (Fast mode: {fast_mode}, Analysis mode: {analysis_mode})\n")
    
    # Setup logging
    log_path = f"{OUTPUT_PATH}/model_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    logger.info(f"Starting execution with fast_mode={fast_mode}, feature_selection={feature_selection}, analysis_mode={analysis_mode}")
    
    # Create analysis directory
    analysis_path = f"{OUTPUT_PATH}/analysis"
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    try:
        # Check if preprocessed data exists
        preprocessed_data_path = f"{OUTPUT_PATH}/preprocessed_features"
        
        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)
        
        # Check if we can skip data preprocessing
        if (os.path.exists(f"{preprocessed_data_path}/train_features.pkl") and
            os.path.exists(f"{preprocessed_data_path}/test_features.pkl") and
            os.path.exists(f"{preprocessed_data_path}/train_target.pkl") and
            os.path.exists(f"{preprocessed_data_path}/train_id.pkl") and
            os.path.exists(f"{preprocessed_data_path}/test_id.pkl") and
            os.path.exists(f"{preprocessed_data_path}/feat_cols.pkl")):
            
            logger.info("Loading preprocessed features from disk...")
            train_features = pd.read_pickle(f"{preprocessed_data_path}/train_features.pkl")
            test_features = pd.read_pickle(f"{preprocessed_data_path}/test_features.pkl")
            train_target = pd.read_pickle(f"{preprocessed_data_path}/train_target.pkl")
            train_id = pd.read_pickle(f"{preprocessed_data_path}/train_id.pkl")
            test_id = pd.read_pickle(f"{preprocessed_data_path}/test_id.pkl")
            feat_cols = pd.read_pickle(f"{preprocessed_data_path}/feat_cols.pkl")
            
        else:
            logger.info("Processing data from scratch...")
            # Load data
            train_df, test_df, bureau, bureau_balance, prev, pos, installments, cc_balance = load_data()
            
            # Process application data
            train_df, test_df, train_target, train_id, test_id = process_application_data(train_df, test_df)
            train_df = optimize_dataframe(train_df)
            test_df = optimize_dataframe(test_df)
            
            # Process bureau data - with caching
            bureau_file = f"{preprocessed_data_path}/bureau_features.pkl"
            if os.path.exists(bureau_file):
                bureau_features = pd.read_pickle(bureau_file)
            else:
                bureau_features = process_bureau_data(bureau, bureau_balance, train_id, test_id)
                bureau_features = optimize_dataframe(bureau_features)
                bureau_features.to_pickle(bureau_file)
            
            # Process previous applications - with caching
            prev_file = f"{preprocessed_data_path}/prev_features.pkl"
            if os.path.exists(prev_file):
                prev_features = pd.read_pickle(prev_file)
            else:
                prev_features = process_previous_applications(prev, train_id, test_id)
                prev_features = optimize_dataframe(prev_features)
                prev_features.to_pickle(prev_file)
            
            # Process POS cash balance - with caching
            pos_file = f"{preprocessed_data_path}/pos_features.pkl"
            if os.path.exists(pos_file):
                pos_features = pd.read_pickle(pos_file)
            else:
                pos_features = process_pos_cash(pos, train_id, test_id)
                pos_features = optimize_dataframe(pos_features)
                pos_features.to_pickle(pos_file)
            
            # Process installments payments - with caching
            inst_file = f"{preprocessed_data_path}/inst_features.pkl"
            if os.path.exists(inst_file):
                inst_features = pd.read_pickle(inst_file)
            else:
                inst_features = process_installments(installments, train_id, test_id)
                inst_features = optimize_dataframe(inst_features)
                inst_features.to_pickle(inst_file)
            
            # Process credit card balance - with caching
            cc_file = f"{preprocessed_data_path}/cc_features.pkl"
            if os.path.exists(cc_file):
                cc_features = pd.read_pickle(cc_file)
            else:
                cc_features = process_credit_card(cc_balance, train_id, test_id)
                cc_features = optimize_dataframe(cc_features)
                cc_features.to_pickle(cc_file)
            
            # Combine all features
            train_features, test_features, feat_cols, train_target, train_id, test_id = combine_features(
                train_df, test_df, bureau_features, prev_features, pos_features, inst_features, cc_features,
                train_target, train_id, test_id
            )
            
            # Save preprocessed features
            logger.info("Saving preprocessed features to disk...")
            train_features.to_pickle(f"{preprocessed_data_path}/train_features.pkl")
            test_features.to_pickle(f"{preprocessed_data_path}/test_features.pkl")
            train_target.to_pickle(f"{preprocessed_data_path}/train_target.pkl")
            train_id.to_pickle(f"{preprocessed_data_path}/train_id.pkl")
            test_id.to_pickle(f"{preprocessed_data_path}/test_id.pkl")
            pd.Series(feat_cols).to_pickle(f"{preprocessed_data_path}/feat_cols.pkl")
        
        # Feature selection if enabled
        if feature_selection:
            logger.info(f"Performing feature selection to get top {n_features} features")
            selected_file = f"{preprocessed_data_path}/selected_features_{n_features}.pkl"
            
            if os.path.exists(selected_file):
                feat_cols = pd.read_pickle(selected_file)
                logger.info(f"Loaded {len(feat_cols)} selected features from disk")
            else:
                feat_cols = select_features(train_features, train_target, feat_cols, n_features)
                pd.Series(feat_cols).to_pickle(selected_file)
        
        # Store all model predictions for analysis
        all_oof_preds = {}
        all_test_preds = {}
        
        # Train models
        logger.info("Training models...")
        
        # LGBM model
        lgbm_submission, lgbm_importance, lgbm_oof = model_training_lgbm(
            train_features, test_features, feat_cols, train_target, train_id, test_id, 
            folds=5, fast_mode=fast_mode
        )
        all_oof_preds['lgbm'] = lgbm_oof
        all_test_preds['lgbm'] = lgbm_submission['TARGET'].values
        
        # Save feature importance for analysis
        if not lgbm_importance.empty:
            lgbm_importance.to_csv(f"{analysis_path}/lgbm_feature_importance.csv", index=False)
        
        # If not in fast mode, train more models
        submissions_to_blend = [lgbm_submission['TARGET']]
        
        if not fast_mode:
            # XGB model
            try:
                xgb_submission, xgb_importance, xgb_oof = model_training_xgb(
                    train_features, test_features, feat_cols, train_target, train_id, test_id,
                    folds=5, fast_mode=fast_mode
                )
                submissions_to_blend.append(xgb_submission['TARGET'])
                all_oof_preds['xgb'] = xgb_oof
                all_test_preds['xgb'] = xgb_submission['TARGET'].values
                
                # Save feature importance
                if not xgb_importance.empty:
                    xgb_importance.to_csv(f"{analysis_path}/xgb_feature_importance.csv", index=False)
                
            except Exception as e:
                logger.error(f"Error in XGBoost training: {e}")
                
            # CatBoost model
            try:
                cat_submission, cat_importance, cat_oof = model_training_catboost(
                    train_features, test_features, feat_cols, train_target, train_id, test_id,
                    folds=5, fast_mode=fast_mode
                )
                submissions_to_blend.append(cat_submission['TARGET'])
                all_oof_preds['catboost'] = cat_oof
                all_test_preds['catboost'] = cat_submission['TARGET'].values
                
                # Save feature importance
                if not cat_importance.empty:
                    cat_importance.to_csv(f"{analysis_path}/catboost_feature_importance.csv", index=False)
                
            except Exception as e:
                logger.error(f"Error in CatBoost training: {e}")
        
        # Blend models if we have more than one
        if len(submissions_to_blend) > 1:
            blend_submission = blend_models(submissions_to_blend, test_id)
            all_test_preds['blend'] = blend_submission['TARGET'].values
            
            # Calculate blended OOF predictions
            blend_oof = np.zeros_like(all_oof_preds['lgbm'])
            for model_name in all_oof_preds:
                blend_oof += all_oof_preds[model_name] / len(all_oof_preds)
            all_oof_preds['blend'] = blend_oof
        
        # Calculate OOF AUC for each model
        logger.info("Model performance summary:")
        for model_name, preds in all_oof_preds.items():
            logger.info(f"{model_name.upper()} OOF AUC: {roc_auc_score(train_target, preds):.6f}")
        
        # Run additional analysis if in analysis mode
        if analysis_mode:
            logger.info("Running additional model analysis...")
            
            # Plot ROC curves
            plot_roc_curves(all_oof_preds, train_target, analysis_path)
            
            # Analyze feature importance (using LightGBM importance as it's always available)
            if not lgbm_importance.empty:
                feature_analysis = analyze_feature_importance(
                    lgbm_importance, train_features, train_target, analysis_path
                )
                
            # Analyze feature correlations
            feature_correlations = analyze_feature_correlations(
                train_features, train_target, n_features=30, output_path=analysis_path
            )
            
            # Compare model predictions
            if len(all_oof_preds) > 1:
                model_comparison = compare_model_predictions(
                    all_oof_preds, train_features, train_target, test_features, test_id, analysis_path
                )
            
            # Analyze model performance by segment
            segment_performance = analyze_model_performance(
                all_oof_preds, train_features, test_features, train_target, test_id, analysis_path
            )
            
            # Save all model predictions for further analysis
            oof_df = pd.DataFrame({'SK_ID_CURR': train_id.values, 'TARGET': train_target.values})
            test_df = pd.DataFrame({'SK_ID_CURR': test_id.values})
            
            for model_name, preds in all_oof_preds.items():
                oof_df[f'{model_name}_pred'] = preds
                
            for model_name, preds in all_test_preds.items():
                test_df[f'{model_name}_pred'] = preds
                
            oof_df.to_csv(f"{analysis_path}/all_oof_predictions.csv", index=False)
            test_df.to_csv(f"{analysis_path}/all_test_predictions.csv", index=False)
            
            logger.info("Analysis complete! Results saved to analysis directory.")
        
        logger.info("\nSubmission files created in the output directory!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Solution')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode with reduced parameters')
    parser.add_argument('--feature_selection', action='store_true', help='Use feature selection')
    parser.add_argument('--n_features', type=int, default=200, help='Number of features to select')
    parser.add_argument('--analyze', action='store_true', help='Run additional analysis')
    
    args = parser.parse_args()
    
    main(fast_mode=args.fast, feature_selection=args.feature_selection, 
         n_features=args.n_features, analysis_mode=args.analyze)