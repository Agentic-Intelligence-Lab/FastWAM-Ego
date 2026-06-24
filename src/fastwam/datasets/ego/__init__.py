"""EgoVerse zarr datasets for FastWAM video-only augmentation."""

from .ego_zarr_reader import EgoZarrEpisode

__all__ = ["EgoZarrEpisode", "EgoVerseVideoDataset"]


def __getattr__(name: str):
    if name == "EgoVerseVideoDataset":
        from .ego_verse_video_dataset import EgoVerseVideoDataset

        return EgoVerseVideoDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
