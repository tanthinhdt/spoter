import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from mediapipe.python.solutions import holistic


HAND_IDENTIFIERS = {
    0: "wrist",
    1: "thumbCMC",
    2: "thumbMP",
    3: "thumbIP",
    4: "thumbTip",
    5: "indexMCP",
    6: "indexPIP",
    7: "indexDIP",
    8: "indexTip",
    9: "middleMCP",
    10: "middlePIP",
    11: "middleDIP",
    12: "middleTip",
    13: "ringMCP",
    14: "ringPIP",
    15: "ringDIP",
    16: "ringTip",
    17: "littleMCP",
    18: "littlePIP",
    19: "littleDIP",
    20: "littleTip",
}
BODY_IDENTIFIERS = {
    0: "nose",
    2: "leftEye",
    5: "rightEye",
    7: "leftEar",
    8: "rightEar",
    11: "leftShoulder",
    12: "rightShoulder",
    13: "leftElbow",
    14: "rightElbow",
    15: "leftWrist",
    16: "rightWrist",
}
VAL_SIGNER_IDS = [1, 2]
TEST_SIGNER_IDS = [7]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracts the keypoints from a dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The path to the dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The split of the dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="The threshold for the visibility of the keypoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The path to the output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the existing files",
    )
    return parser.parse_args()


def extract_keypoints(
    keypoints_detector, video_path: str, threshold: float = 0.0
) -> dict:
    video_data = dict()
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = keypoints_detector.process(frame)

        for idx, landmark_name in BODY_IDENTIFIERS.items():
            x, y = 0.0, 0.0
            if results.pose_landmarks is not None:
                keypoint = results.pose_landmarks.landmark[idx]
                if keypoint.visibility >= threshold:
                    x, y = keypoint.x, keypoint.y
            x_key = landmark_name + "_X"
            y_key = landmark_name + "_Y"
            video_data[x_key] = video_data.get(x_key, []) + [x]
            video_data[y_key] = video_data.get(y_key, []) + [y]

        for idx, landmark_name in HAND_IDENTIFIERS.items():
            x, y = 0.0, 0.0
            if results.left_hand_landmarks is not None:
                keypoint = results.left_hand_landmarks.landmark[idx]
                if keypoint.visibility >= threshold:
                    x, y = keypoint.x, keypoint.y
            if landmark_name.startswith("left"):
                x_key, y_key = landmark_name + "_X", landmark_name + "_Y"
            else:
                x_key, y_key = landmark_name + "_left_X", landmark_name + "_left_Y"
            video_data[x_key] = video_data.get(x_key, []) + [x]
            video_data[y_key] = video_data.get(y_key, []) + [y]

        for idx, landmark_name in HAND_IDENTIFIERS.items():
            x, y = 0.0, 0.0
            if results.right_hand_landmarks is not None:
                keypoint = results.right_hand_landmarks.landmark[idx]
                if keypoint.visibility >= threshold:
                    x, y = keypoint.x, keypoint.y
            if landmark_name.startswith("right"):
                x_key, y_key = landmark_name + "_X", landmark_name + "_Y"
            else:
                x_key, y_key = landmark_name + "_right_X", landmark_name + "_right_Y"
            video_data[x_key] = video_data.get(x_key, []) + [x]
            video_data[y_key] = video_data.get(y_key, []) + [y]
    cap.release()

    assert len(video_data) == len(BODY_IDENTIFIERS) * 2 + len(HAND_IDENTIFIERS) * 4, (
        f"Expected {len(BODY_IDENTIFIERS) * 2 + len(HAND_IDENTIFIERS) * 4} landmarks, "
        f"but got {len(video_data)} landmarks"
    )
    for key, value in video_data.items():
        assert (
            len(value) == num_frames
        ), f"Expected {num_frames} frames, but got {len(value)} frames for {key}"
        video_data[key] = str(value)

    return video_data


def main(args: argparse.Namespace) -> None:
    keypoints_detector = holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
    )
    video_dir = os.path.join(args.data_dir, "rgb_videos")
    meta_df = pd.read_json(
        os.path.join(args.data_dir, "meta.json"),
        dtype={
            "video_id": "string",
            "signer_id": "int",
        },
    )
    gloss2id = pd.read_csv(
        os.path.join(args.data_dir, "gloss.csv"),
        names=["id", "gloss"],
        index_col="gloss"
    ).to_dict()["id"]
    if args.split == "train":
        meta_df = meta_df[~meta_df["signer_id"].isin(VAL_SIGNER_IDS + TEST_SIGNER_IDS)]
    elif args.split == "val":
        meta_df = meta_df[meta_df["signer_id"].isin(VAL_SIGNER_IDS)]
    elif args.split == "test":
        meta_df = meta_df[meta_df["signer_id"].isin(TEST_SIGNER_IDS)]
    else:
        raise ValueError("Invalid split")
    meta_df = meta_df.sample(frac=1).reset_index(drop=True)
    output_path = os.path.join(args.output_dir, f"{args.split}.csv")
    if os.path.exists(output_path) and not args.overwrite:
        df = pd.read_csv(output_path)
    else:
        df = None
    os.makedirs(args.output_dir, exist_ok=True)

    for row in tqdm(meta_df.itertuples(), total=len(meta_df)):
        video_path = os.path.join(video_dir, f"{row.video_id}.mp4")
        video_data = extract_keypoints(keypoints_detector, video_path, args.threshold)
        video_data["labels"] = video_data.get("labels", []) + [gloss2id[row.gloss]]

        if df is None:
            df = pd.DataFrame(video_data)
        else:
            df = pd.concat([df, pd.DataFrame(video_data)], ignore_index=True)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = get_args()
    main(args)
