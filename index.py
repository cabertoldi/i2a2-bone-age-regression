import pandas as pd

from src.prepareimages import normalize_images
from src.preparedatasets import organize_datasets
from src.neuralnetwork import init_model, prepare_dataset

def main():
    normalize_images('train')
    organize_datasets()

    df_submission = pd.read_csv('./data/sample_submission.csv')
    normalize_images('sample_submission')

    results = _results(df_submission)
    results.to_csv("results.csv",index=False)

def _results(df):
    girls = _run_model('F', df)
    boys = _run_model('M', df)

    return pd.concat([girls, boys], ignore_index=True)

def _run_model(patientSex, df):
    df_submission = df.query(f'patientSex == "{patientSex}"')
    X, Y = prepare_dataset(df_submission)

    X = X.astype('float32')/255

    max_bornage = Y.max()
    Y = Y / max_bornage

    model = init_model(f'{patientSex}', df_submission)

    predicted = model.predict_generator(X)
    predicted_months = max_bornages*(predicted.flatten())

    filenames = df_submission['fileName']
    return pd.DataFrame({ "fileName": filenames, "boneage": predicted_months })

if __name__ == '__main__':
    main()