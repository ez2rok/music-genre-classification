# import statements
import os
import librosa
import math
import json


# global variables
DATASET_PATH = "GTZAN-genre-dataset"
JSON_PATH = "inputs.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_genre_label(root, data):
    """ save the genre label to the mapping """

    root_components = root.split("/")  # genres/blues => ["genre", "blues"]
    semantic_label = root_components[-1]
    data["mapping"].append(semantic_label)
    print("\nProcessing: {}".format(semantic_label))
    return data


def get_mfcc_of_segment(s, samples_per_segment, signal, n_mfcc, n_fft, hop_length):
    """ given a segment of an audiofile, return the mfcc for that segment """

    # compute start and end of signal
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


def append_mfcc_to_data(s, i, mfcc, num_mfcc_vectors_per_segment, data, file_path):
    """ store only MFCC features with the expected number of vectors """

    if len(mfcc) == num_mfcc_vectors_per_segment:
        data["mfcc"].append(mfcc.tolist())
        data["labels"].append(i-1)
        print("{}, segment:{}".format(file_path, s+1))


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048,
              hop_length=512, num_segments=10):
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

    # define values
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],  # eg: classical is mapped to 0, blues to 1, etc.
        "mfcc": [],
        "labels": [],
    }

    # loop through all the genre directories
    for i, (root, _, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we are processing a genre sub-directory
        if root is not dataset_path:
            if root.split("/")[-1] == "metal":
                continue

            # save the genre label to the mapping
            data = save_genre_label(root, data)

            # process audiofiles for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(root, f)
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                # compute MFCC for each segment
                for s in range(num_segments):
                    mfcc = get_mfcc_of_segment(s, samples_per_segment, signal,
                                               n_mfcc, n_fft, hop_length)
                    append_mfcc_to_data(s, i, mfcc, num_mfcc_vectors_per_segment,
                                        data, file_path)

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
