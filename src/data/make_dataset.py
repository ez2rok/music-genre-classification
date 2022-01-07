import os


def download_dataset(data_dir, raw_data_path):
    """
    download the gtzan genre dataset 
    """

    # initial values
    print('Downloading GTZAN genre dataset...')
    url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
    os.system('mkdir -p {}'.format(data_dir))
    temp_filepath = data_dir + '/temporary_genre_dataset.tar.gz'

    # download .tar.gz dataset, unzip it, rename it, and delete unnecessary .mf files
    os.system('wget {} -O {}'.format(url, temp_filepath))
    os.system('tar -vxzf {}'.format(temp_filepath))
    os.system('mv genres {}'.format(raw_data_path))
    os.system('rm -rf {}/*.mf'.format(raw_data_path))
    os.system('rm {}'.format(temp_filepath))
    os.system('rm -rf genres')
    print('Finished downloading GTZAN genre dataset')
