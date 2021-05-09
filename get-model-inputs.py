# import statements
import os
import librosa
import math
import json


# global variables
DATASET_PATH = "GTZAN-genre-dataset"
JSON_PATH = "model-inputs.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def get_mfcc_segment_values(num_segments, hop_length):
    """ compute values associated with the mfcc of each song segment"""

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    return samples_per_segment, num_mfcc_vectors_per_segment


def not_a_genre_directory(root, dataset_path):
    """ determine if the root directory is a genre directory
    All directories except for the dataset_path are genre directories """

    return root is dataset_path


def save_genre_name_to_data(root, data):
    """ save the name of the genre to the data dictionary """

    root_components = root.split("/")
    genre_name = root_components[-1]
    data["mapping"].append(genre_name)
    print("\nProcessing: {}".format(genre_name))


def get_file_path(root, f):
    """ get the path to a file """

    file_path = os.path.join(root, f)
    return file_path


def get_audio_signal(file_path):
    """ get the audio signal of a file """

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return signal


def get_mfcc(signal, samples_per_segment, n_mfcc, n_fft, hop_length, s):
    """ compute the mfcc of a song segment """

    # compute the start and end of the song segment
    start = samples_per_segment * s
    finish = start + samples_per_segment

    # compute mfcc for the segment
    mfcc = librosa.feature.mfcc(signal[start: finish],
                                sr=SAMPLE_RATE,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length
                                )
    mfcc = mfcc.T
    return mfcc


def necessary_to_save_mfcc(mfcc, num_mfcc_vectors_per_segment):
    """ only save mfccs whose length equals num_mfcc_vectors_per_segment

    ---- Explanation ----
    A song is divided into x=num_segments number of segments.
    An mfcc matrix concatenates all the mfcc vectors present in the current segment.
    Each mfcc vector is computed on hop-length number of samples.

    num_mfcc_vectors_per_segment is the number of mfcc vectors we can compute in a 
    given song segment. This is the length of the mfcc matrix returned by get_mfcc(). 
    If the segment of a song contains a different amount of mfcc vectors, the mfcc 
    matrix will have a different width and we cannot use it. We thus only save
    mfccs with the same length, ie a length equal to num_mfcc_vectors_per_segment.
    """

    return len(mfcc) == num_mfcc_vectors_per_segment


def save_mfcc_to_data(mfcc, data, file_path, i, s,):
    """ append the mfcc and the (encoded) genre it came from to the data dictionary  """

    data["mfcc"].append(mfcc.tolist())
    data["labels"].append(i-1)
    print("{}, segment:{}".format(file_path, s+1))


def write_data_to_json_file(data, json_path):
    """ save the data dictionary to a json file """

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def get_model_inputs(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """ extract MFCCs from music dataset and saves them into a json file with genre labels

    ---- Parameters ----
    dataset_path (str): Path to dataset
    json_path (str): Path to json file used to save MFCCs
    n_mfcc (int): Number of coefficents to extract
    n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    hop_length (int): Sliding window for FFT. Measured in # of samples
    num_segments (int): Number of segments we want to divide sample tracks into

    ---- Return ----
    None
    """

    # these values are used in necessary_to_save_mfcc()
    # we compute them now because we do not want to compute them multiple times in the for loop
    samples_per_segment, num_mfcc_vectors_per_segment = get_mfcc_segment_values(
        num_segments, hop_length)

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],  # eg: classical is mapped to 0, blues to 1, etc.
        "mfcc": [],
        "labels": [],
    }

    for i, (root, _, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we are processing a genre directory
        # and save the name of the genre to the data dictionary
        if not_a_genre_directory(root, dataset_path):
            continue
        save_genre_name_to_data(root, data)

        # process all songs in the current genre directory
        for f in filenames:
            # load the audio file
            file_path = get_file_path(root, f)
            signal = get_audio_signal(file_path)

            # divide the song into segments; compute and save the mfcc of each segment
            for s in range(num_segments):
                mfcc = get_mfcc(signal, samples_per_segment,
                                n_mfcc, n_fft, hop_length, s)
                if necessary_to_save_mfcc(mfcc, num_mfcc_vectors_per_segment):
                    save_mfcc_to_data(mfcc, data, file_path, i, s,)

    # save the data dictionary to a json file
    write_data_to_json_file(data, json_path)


if __name__ == "__main__":
    
    # this takes approximately five minutes to run
    get_model_inputs(DATASET_PATH, JSON_PATH, num_segments=10)
