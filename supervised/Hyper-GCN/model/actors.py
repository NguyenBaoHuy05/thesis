"""Real-actor mask of a preprocessed NTU skeleton batch."""


def valid_actor_mask(x, atol=1e-6):
    """x: [N, C, T, V, M] skeleton after feeder preprocessing. An actor is valid when it is a
    real person: not all-zero, and (for m > 0) not a copy of the first actor, which is how
    align_frames fills the second slot of a single-person sample. The feeder transforms every
    actor identically (same rotation, crop/resize, centering, bone/velocity), so a copy stays
    a copy and an all-zero actor stays all-zero."""
    nonzero = x.abs().amax(dim=(1, 2, 3)) > atol
    copy = (x - x[..., :1]).abs().amax(dim=(1, 2, 3)) <= atol
    copy[:, 0] = False
    return nonzero & ~copy
