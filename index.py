from prepareimages import normalize_images
from preparedatasets import organize_train_datasets
from neuralnetwork import init_model

def main():
    normalize_images('train')
    organize_train_datasets()
    
    # normalize_images('test')
    # init_model('F')

if __name__ == '__main__':
    main()