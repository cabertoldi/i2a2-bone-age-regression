import cv2
import pandas as pd

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def organize_train_datasets():
    df_train = pd.read_csv('./data/train.csv')

    df_train_female = df_train.query('patientSex == "F"')
    first_group_F, second_group_F, third_group_F, fourth_group_F, last_group_F = _extract_groups(df_train_female)
    
    df_train_male = df_train.query('patientSex == "M"')
    first_group_M, second_group_M, third_group_M, fourth_group_M, last_group_M = _extract_groups(df_train_male)

    first_increased_group_F = _update_dataset(first_group_F, 'F', 4).head(1000)
    second_group_F = second_group_F.sample(n=1000, random_state=1)
    third_group_F = third_group_F.sample(n=1000, random_state=1)
    fourth_group_F = fourth_group_F.sample(n=1000, random_state=1)

    train_final_F = pd.concat([first_increased_group_F, second_group_F, third_group_F, fourth_group_F], ignore_index=True)
    train_final_F.to_csv('./data/F-train.csv')

    first_increased_group_M = _update_dataset(first_group_F, 'F', 4).head(1000)
    second_group_M = second_group_M.sample(n=1000, random_state=1)
    third_group_M = third_group_M.sample(n=1000, random_state=1)
    fourth_group_M = fourth_group_M.sample(n=1000, random_state=1)
    last_increased_group_M = _update_dataset(last_group_M, 'M', 4).head(1000)

    train_final_F = pd.concat([first_increased_group_F, second_group_F, third_group_F, fourth_group_F], ignore_index=True)
    train_final_F.to_csv('./data/M-train.csv')

def _extract_groups(df_train):
    first_group = df_train.query('boneage < 50')
    print('Less than 50 months: ', len(first_group))

    second_group = df_train.query('boneage >= 50 and boneage < 100')
    print('Between 50 and 100 months: ', len(second_group))

    third_group = df_train.query('boneage >= 100 and boneage < 150')
    print('Between 100 and 150 months: ', len(third_group))

    fourth_group = df_train.query('boneage >= 150 and boneage < 200')
    print('Between 150 and 200 months: ', len(fourth_group))

    last_group = df_train.query('boneage >= 200')
    print('More than 200 months: ', len(last_group))

    return first_group, second_group, third_group, fourth_group, last_group

def _image_generated(filename, number_samples):
    image = load_img(f"./data/clean-images-train/{filename}")
    data = img_to_array(image)

    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(width_shift_range=[-200,200])

    iterator = datagen.flow(samples, batch_size=1)
    new_images = []
    for i in range(number_samples):
        batch = iterator.next()
        image_generated = batch[0].astype('uint8')
        new_images.append(image_generated)

    return new_images

def _update_dataset(dataset, patientSex, number_samples):
    mean_bornage = dataset['boneage'].mean()

    for filename in dataset['fileName']:
        new_images = _image_generated(filename, number_samples)
        for idx, image in enumerate(new_images):
            cv2.imwrite(f"./data/clean-images-train/{idx}-{filename}", image)

            new_data = pd.DataFrame({"fileName":[f"{idx}-{filename}"], "patientSex":[patientSex], "boneage": [mean_bornage]})
            dataset = pd.concat([dataset, new_data], ignore_index=True)

    return dataset