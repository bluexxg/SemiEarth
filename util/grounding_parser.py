"""Parse Qwen grounding records without inventing incomplete coordinates."""
import json
import math
import re


def parse_grounding(text, class_names):
    names = {name.casefold(): name for name in class_names}
    result = {}

    def add(label, box):
        if not isinstance(label, str) or label.casefold() not in names:
            return
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            return
        if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
            return
        name = names[label.casefold()]
        coords = tuple(box)
        if coords not in result.setdefault(name, []):
            result[name].append(coords)

    # Decode only complete JSON objects, including those preceding a truncated tail.
    # raw_decode respects JSON strings/escapes and never synthesizes closing tokens.
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find('{', pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except (ValueError, json.JSONDecodeError):
            pos = start + 1
            continue
        if isinstance(obj, dict):
            label = obj.get('class_name', obj.get('label'))
            other = obj.get('label')
            if not (isinstance(label, str) and isinstance(other, str) and label.casefold() != other.casefold()):
                add(label, obj.get('bbox_2d'))
        pos = end

    # Preserve the original plain-text format; match a whole class name.
    for name in class_names:
        pattern = rf'(?<![\w]){re.escape(name)}[:\s]*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        for match in re.findall(pattern, text, re.IGNORECASE):
            add(name, list(map(int, match)))
    return result
