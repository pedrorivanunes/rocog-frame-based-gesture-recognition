"""Measure whether a pose detector can see the people in this dataset.

Body keypoints are invariant to domain by construction: an elbow is an elbow
whether it was rendered or filmed. That makes a skeleton representation the one
candidate in this project that could remove the synthetic-to-real gap rather
than narrow it, and it is why the question is worth asking before a day is
spent building the branch that would use it.

The risk sits in exactly the place the gap does. The dominant variation in the
real footage is lighting, and some clips were shot under cloud heavy enough to
leave the subject close to a silhouette. A keypoint detector given a silhouette
does not report that it is lost: it returns a full skeleton with low confidence
scores, and a pipeline that ignores those scores trains on invented joints.

So this probe reads the confidence rather than the coordinates. It asks two
things of a small sample of frames: whether the detector fired at all, and how
sure it is about the joints this task actually needs. Both are reported against
a lighting measure taken from the frames themselves, which is what turns a
guess about dark clips into a number.

The probe has no bad outcome. A detector that copes clears the way for the
skeleton branch; one that fails yields the failure rate per domain and per
lighting level, which is a property of this dataset that no publication about
it reports.

Run it in an environment of its own. MediaPipe pins dependencies that the ROCm
build of torch also pins, and the training environment is not worth risking for
a measurement that writes a CSV and exits.
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd

from splits import sample_videos

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The joints the seven gestures are made of. All of them are commands signalled
# with the arms, so a detector that places the ankles perfectly and loses the
# wrists has failed at this task while scoring well on average over the
# thirty-three points a full skeleton carries. Indices are the BlazePose
# topology and are checked against the installed enum in ``make_detector``,
# because a constant that silently stops meaning what it says would move every
# number this probe reports.
ARM_LANDMARK_NAMES = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
)
ARM_LANDMARKS = (11, 12, 13, 14, 15, 16)

# Confidence below which a joint counts as guessed rather than located. The
# detector reports a score for every landmark whenever it fires, including for
# joints it could not see, so a threshold is the only thing separating the two.
# Set where it is because the probe reports the median alongside the count:
# if the choice matters, the medians will say so.
VISIBILITY_FLOOR = 0.5

# How many frames to read from each video. Lighting is a property of a
# recording, not of a frame, so breadth across videos buys more than depth
# within one.
FRAMES_PER_VIDEO = 4

# Number of lighting groups the videos are split into for reporting. Quartiles
# over the videos present, not fixed brightness thresholds: the question is
# whether the dark end of this dataset behaves differently from the bright end,
# and where those ends sit is a fact about the dataset rather than an input.
LIGHTING_GROUPS = 4


class Lighting(NamedTuple):
    """How well lit a frame is.

    Attributes:
        level: Mean perceived brightness, 0 to 255.
        spread: Standard deviation of the same quantity. A frame can be dark
            overall and still show a person clearly; it is the contrast between
            subject and background that a detector needs, and a silhouette
            against bright cloud scores low on ``level`` and high on ``spread``.
    """

    level: float
    spread: float


class Visibility(NamedTuple):
    """What the detector claims to have seen, over a chosen set of joints.

    Attributes:
        located: How many of those joints scored at or above the floor.
        median: Median confidence over them, floor or not. Reported beside the
            count so that a run can tell a clean miss from a set of joints
            sitting just under the threshold.
    """

    located: int
    median: float


def luminance(frame: np.ndarray) -> Lighting:
    """Measure how well lit one frame is.

    Weighted by how much each channel contributes to perceived brightness
    rather than averaged flat, because a flat mean calls a saturated green
    field as bright as the sky above it, and this dataset is mostly field.

    The real clips arrive already cropped around the subject, so the whole
    frame is a fair proxy for how lit the person is. That is inherited from the
    dataset's own preprocessing, not a property of the footage, and it would
    stop holding on uncropped video.

    Args:
        frame: One frame, BGR as OpenCV reads it.

    Returns:
        The lighting of that frame.
    """
    blue, green, red = (frame[:, :, channel].astype(np.float64) for channel in range(3))
    perceived = 0.114 * blue + 0.587 * green + 0.299 * red

    return Lighting(float(perceived.mean()), float(perceived.std()))


def visibility_summary(
    visibilities: list[float],
    indices: tuple[int, ...] = ARM_LANDMARKS,
    floor: float = VISIBILITY_FLOOR,
) -> Visibility:
    """Summarise the detector's confidence over the joints this task needs.

    Args:
        visibilities: One score per landmark, in the detector's own order.
        indices: Which landmarks to summarise over.
        floor: Score at or above which a joint counts as located.

    Returns:
        The count located and the median score.

    Raises:
        IndexError: If an index falls outside the scores given. Raised rather
            than skipped, since a short list means the detector returned a
            different topology than the constants describe.
    """
    chosen = [visibilities[index] for index in indices]

    return Visibility(
        located=sum(score >= floor for score in chosen),
        median=float(np.median(chosen)),
    )


def spread_indices(count: int, wanted: int) -> list[int]:
    """Choose evenly spaced positions out of a run of rows.

    The centre of each of ``wanted`` equal segments, which is the rule
    evaluation already uses to pick frames from a video. Taking the first few
    rows instead would read only the opening of every gesture, where the body
    is still close to neutral, and a detector's job is easiest there.

    Args:
        count: How many rows there are.
        wanted: How many to keep.

    Returns:
        Positions to keep, ascending. Asking for at least as many as exist
        returns all of them.
    """
    if wanted >= count:
        return list(range(count))

    segment = count / wanted

    return [int(segment * index + segment / 2) for index in range(wanted)]


def lighting_groups(levels: pd.Series, groups: int = LIGHTING_GROUPS) -> pd.Series:
    """Rank videos into lighting groups, darkest first.

    By rank rather than by value, so that each group holds a comparable number
    of videos. Cutting on brightness itself would be the more natural reading
    but leaves the darkest group holding whatever handful of clips sit in the
    tail, and a detection rate over four videos says nothing.

    Args:
        levels: One brightness value per video, indexed by video.
        groups: How many groups to form.

    Returns:
        A group number per video, 0 for the darkest. Ties are broken by order,
        which keeps the groups balanced when many videos share a value.
    """
    order = levels.rank(method="first")

    return ((order - 1) * groups // len(levels)).astype(int)


def make_detector():
    """Build the pose detector, and check that it is the one assumed here.

    Imported inside the function so that everything above can be tested, and
    this module imported, in the training environment, which has no MediaPipe
    in it and should not acquire one.

    Static mode on purpose. The frames a manifest lists are scattered across a
    video rather than consecutive, so there is no motion for a tracker to
    follow; running full detection on every frame is both the honest setting
    and the slower one.

    Returns:
        A detector ready to be called on RGB frames.

    Raises:
        RuntimeError: If the installed landmark numbering disagrees with
            ``ARM_LANDMARKS``. The probe's whole output is the confidence of
            six specific joints, and reading six wrong ones would produce a
            plausible table about the ankles.
    """
    import mediapipe as mp

    landmark = mp.solutions.pose.PoseLandmark
    for index, name in zip(ARM_LANDMARKS, ARM_LANDMARK_NAMES, strict=True):
        if landmark(index).name != name:
            raise RuntimeError(
                f"landmark {index} is {landmark(index).name}, expected {name}"
            )

    return mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)


def landmark_visibilities(detector, frame: np.ndarray) -> list[float] | None:
    """Run the detector on one frame and return its per-joint confidence.

    Args:
        detector: As built by ``make_detector``.
        frame: One frame, BGR as OpenCV reads it. Converted here, because the
            detector expects RGB and handing it BGR swaps the red and blue
            channels — which does not fail, it just quietly makes every face
            and every hand the wrong colour.

    Returns:
        One score per landmark, or ``None`` if the detector found no person.
    """
    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return None

    return [point.visibility for point in result.pose_landmarks.landmark]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which manifests to probe, and how densely.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``manifest``, ``fraction``, ``frames_per_video``,
        ``seed`` and ``floor``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        nargs="+",
        required=True,
        help="manifests to probe, file names under data/manifests/",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="share of each class's videos to read, drawn per class",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=FRAMES_PER_VIDEO,
        help="how many frames to read from each video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="chooses the draw, so a fraction is reproducible",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=VISIBILITY_FLOOR,
        help="confidence at or above which a joint counts as located",
    )
    return parser.parse_args(argv)


def probe_manifest(
    detector,
    manifest: pd.DataFrame,
    frames_per_video: int,
    floor: float,
) -> pd.DataFrame:
    """Read a sample of a manifest's frames and record what the detector saw.

    One row per frame read, written out so that the reading can be repeated
    without running the detector again — the same rule every other measurement
    in this project follows.

    Args:
        detector: As built by ``make_detector``.
        manifest: Rows to probe, one per frame.
        frames_per_video: How many frames to read from each video.
        floor: Confidence at or above which a joint counts as located.

    Returns:
        One row per frame read, carrying the frame's identity, its lighting,
        whether a person was found, and the confidence over the arm joints.
    """
    records = []

    for video_id, rows in manifest.groupby("video_id", sort=False):
        kept = rows.iloc[spread_indices(len(rows), frames_per_video)]

        for row in kept.itertuples():
            frame = cv2.imread(str(PROJECT_ROOT / row.path))
            if frame is None:
                raise FileNotFoundError(f"could not read frame at {row.path}")

            lighting = luminance(frame)
            visibilities = landmark_visibilities(detector, frame)
            arms = (
                visibility_summary(visibilities, floor=floor)
                if visibilities is not None
                else None
            )

            records.append(
                {
                    "video_id": video_id,
                    "group_id": row.group_id,
                    "class_name": row.class_name,
                    "frame_number": row.frame_number,
                    "level": lighting.level,
                    "spread": lighting.spread,
                    "detected": visibilities is not None,
                    "arms_located": arms.located if arms else 0,
                    "arm_visibility": arms.median if arms else float("nan"),
                    "all_visibility": (
                        float(np.median(visibilities)) if visibilities else float("nan")
                    ),
                }
            )

    return pd.DataFrame(records)


def report(name: str, probed: pd.DataFrame, floor: float) -> None:
    """Print what one manifest's probe found.

    Args:
        name: The manifest's name, for the heading.
        probed: As ``probe_manifest`` returns it.
        floor: The threshold that was used, echoed so the table reads alone.
    """
    videos = probed.groupby("video_id").agg(
        detected=("detected", "mean"),
        level=("level", "mean"),
        arm_visibility=("arm_visibility", "median"),
    )
    groups = lighting_groups(videos["level"])

    print(f"\n{name}  —  {len(probed)} frames, {len(videos)} videos")
    print(f"  detection rate            {probed['detected'].mean():.1%}")
    print(f"  arm joints located        {probed['arms_located'].mean():.2f} of 6")
    print(f"  arm confidence, median    {probed['arm_visibility'].median():.3f}")
    print(f"  all joints, median        {probed['all_visibility'].median():.3f}")

    # A mean detection rate hides the shape that matters. Failure spread thinly
    # over every video is noise a later stage can drop; failure concentrated in
    # a few clips is a subset of the dataset the skeleton branch cannot serve,
    # and only the per-video count separates the two.
    lost = (videos["detected"] < 0.5).sum()
    print(f"  videos detected in under half their frames: {lost} of {len(videos)}")

    print(f"  by lighting, darkest first (confidence floor {floor}):")
    print(
        f"    {'group':>5} {'videos':>7} {'brightness':>11} {'detected':>9} {'arms':>7}"
    )
    for group, members in videos.groupby(groups):
        detected = members["detected"].mean()
        arms = members["arm_visibility"].median()
        print(
            f"    {group:>5} {len(members):>7} {members['level'].mean():>11.1f}"
            f" {detected:>8.1%} {arms:>7.3f}"
        )


if __name__ == "__main__":
    args = parse_args()
    detector = make_detector()
    output_directory = PROJECT_ROOT / "data/pose"
    output_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    for name in args.manifest:
        manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / name)
        manifest = sample_videos(manifest, args.fraction, args.seed)

        probed = probe_manifest(detector, manifest, args.frames_per_video, args.floor)
        probed.to_csv(output_directory / f"probe_{Path(name).stem}.csv", index=False)
        report(name, probed, args.floor)

    print(f"\ntime taken: {time.perf_counter() - start_time:.0f} s")
    print(f"tables written to {output_directory}")
