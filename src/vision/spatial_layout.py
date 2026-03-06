"""
Divides the camera frame into left / center / right zones
and assigns each tracked object to a zone.
"""


def get_zone(bbox: list, frame_width: int) -> str:
    """
    Returns spatial zone of object based on bbox center.

    Zones:
        left   → x_center < 33% of frame width
        center → x_center between 33% and 66%
        right  → x_center > 66%
    """
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2

    left_boundary  = frame_width * 0.33
    right_boundary = frame_width * 0.66

    if x_center < left_boundary:
        return "left"
    elif x_center > right_boundary:
        return "right"
    else:
        return "ahead"


def build_spatial_summary(tracks: list, frame_width: int, estimator) -> str:
    """
    Build a natural-language spatial summary of all tracked objects,
    sorted by proximity (closest first) within each zone.

    Args:
        tracks: List of track dicts from ByteTracker
        frame_width: Camera frame width in pixels
        estimator: DistanceEstimator instance

    Returns:
        A spoken summary string. Example:
        "Ahead: person about 2 metres away. To your left: bicycle far away."
    """
    if not tracks:
        return "Path appears clear."

    zones_raw = {"left": [], "ahead": [], "right": []}

    for track in tracks:
        zone = get_zone(track['bbox'], frame_width)
        distance = estimator.estimate(track['class_name'], track['bbox'])
        dist_str = estimator.format_distance(distance)
        label = track['class_name']
        entry = f"{label} {dist_str}".strip()
        zones_raw[zone].append((label, track['bbox'], entry))

    parts = []
    for zone_name, zone_label in [
        ("ahead", "Ahead"),
        ("left", "To your left"),
        ("right", "To your right")
    ]:
        items = zones_raw[zone_name]
        if items:
            # Sort closest first within each zone
            sorted_items = sorted(
                items,
                key=lambda x: estimator.estimate(x[0], x[1]) or 99
            )
            entries = [i[2] for i in sorted_items]
            parts.append(f"{zone_label}: " + ", ".join(entries))

    return ". ".join(parts) + "." if parts else "Path appears clear."
