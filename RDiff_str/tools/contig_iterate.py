import itertools
import re
from typing import List, Optional


_GAP_PATTERN = re.compile(r"^(\d+)(?:-(\d+)|(\+))?$")
_MOTIF_PATTERN = re.compile(r"^[A-Za-z]\d+-\d+$")


def _calculate_segment_length(segment_spec: str) -> int:
    """
    Calculates the length of a segment from its string specifier.
    The specifier is expected to contain a range like 'StartNum-EndNum'
    (e.g., "A3-16", "segment10-20").
    Length = end_residue - start_residue + 1.

    Args:
        segment_spec (str): The string specifier for the segment.

    Returns:
        int: The calculated length of the segment.

    Raises:
        ValueError: If the segment_spec cannot be parsed to find start/end numbers,
                    or if start_residue > end_residue.
    """
    # Search for a pattern like "digits-digits" to find start and end residue numbers.
    match = re.search(r"(\d+)-(\d+)", segment_spec)
    if match:
        start_residue = int(match.group(1))
        end_residue = int(match.group(2))

        if end_residue < start_residue:
            raise ValueError(
                f"In segment specifier '{segment_spec}', end residue ({end_residue}) "
                f"is less than start residue ({start_residue})."
            )
        return end_residue - start_residue + 1
    else:
        raise ValueError(
            f"Cannot parse start and end residues from segment specifier '{segment_spec}'. "
            "Expected a format containing 'StartNum-EndNum', e.g., 'A3-16'."
        )


def _parse_gap_token(token: str) -> Optional[dict]:
    match = _GAP_PATTERN.fullmatch(token)
    if match is None:
        return None

    start = int(match.group(1))
    end = match.group(2)
    has_plus = match.group(3) is not None
    if has_plus:
        return {"type": "open", "min": start}
    if end is not None:
        end = int(end)
        if end < start:
            raise ValueError(f"Gap range '{token}' has end smaller than start.")
        return {"type": "range", "values": list(range(start, end + 1))}
    return {"type": "range", "values": [start]}


def _integer_partitions(total: int, n_parts: int):
    if n_parts == 0:
        if total == 0:
            yield []
        return
    if n_parts == 1:
        yield [total]
        return
    for value in range(total + 1):
        for suffix in _integer_partitions(total - value, n_parts - 1):
            yield [value] + suffix


def expand_contig_template(
        contig_template: str,
        target_length: Optional[int] = None,
        require_total_length: Optional[bool] = None,
) -> List[str]:
    """
    Expands a contig template into concrete contig strings.

    Tokens are separated by "/". Motif tokens use the standard chain+range form,
    such as "A3-16" or "E35-47". Gap tokens support:
      - "30": fixed gap length
      - "30-35": closed gap-length range
      - "8+": minimum gap length, expanded using target_length

    require_total_length is an advanced internal switch. Leave it as None for
    normal use: templates with "N+" gaps expand to exactly target_length, while
    templates without "N+" gaps may be shorter than target_length.
    """
    if not contig_template:
        raise ValueError("contig_template must not be empty.")

    processed = contig_template.strip()
    if processed.startswith("[") and processed.endswith("]"):
        processed = processed[1:-1]

    raw_tokens = [token.strip() for token in processed.split("/")]
    if any(token == "" for token in raw_tokens):
        raise ValueError(f"Contig template '{contig_template}' contains an empty token.")

    parsed_tokens = []
    motif_total_length = 0
    has_open_gap = False
    range_gap_indices = []
    range_gap_options = []

    for token in raw_tokens:
        gap_spec = _parse_gap_token(token)
        if gap_spec is not None:
            if gap_spec["type"] == "open":
                has_open_gap = True
            else:
                range_gap_indices.append(len(parsed_tokens))
                range_gap_options.append(gap_spec["values"])
            parsed_tokens.append({"kind": "gap", "raw": token, **gap_spec})
            continue

        if not _MOTIF_PATTERN.fullmatch(token):
            raise ValueError(
                f"Contig template '{contig_template}' contains a malformed token: '{token}'")
        motif_total_length += _calculate_segment_length(token)
        parsed_tokens.append({"kind": "motif", "value": token})

    if has_open_gap and target_length is None:
        raise ValueError("target_length is required when a contig template contains an open gap like '8+'.")

    if require_total_length is None:
        require_total_length = has_open_gap

    concrete_contigs = []
    range_products = itertools.product(*range_gap_options) if range_gap_options else [()]

    for range_values in range_products:
        tokens = [dict(token) for token in parsed_tokens]
        for token_idx, value in zip(range_gap_indices, range_values):
            tokens[token_idx]["value"] = value

        fixed_gap_length = sum(
            token["value"] for token in tokens
            if token["kind"] == "gap" and token["type"] == "range")
        open_gap_indices = [
            idx for idx, token in enumerate(tokens)
            if token["kind"] == "gap" and token["type"] == "open"
        ]
        min_open_gap_length = sum(tokens[idx]["min"] for idx in open_gap_indices)
        min_total_length = motif_total_length + fixed_gap_length + min_open_gap_length

        if target_length is not None:
            remaining = target_length - min_total_length
            if remaining < 0:
                continue
        else:
            remaining = 0

        if not open_gap_indices:
            if target_length is not None:
                total_length = motif_total_length + fixed_gap_length
                if require_total_length and total_length != target_length:
                    continue
                if not require_total_length and total_length > target_length:
                    continue
            distributions = [[]]
        elif require_total_length:
            distributions = _integer_partitions(remaining, len(open_gap_indices))
        else:
            distributions = itertools.chain.from_iterable(
                _integer_partitions(extra, len(open_gap_indices))
                for extra in range(remaining + 1)
            )

        for distribution in distributions:
            output_tokens = []
            open_values = iter(distribution)
            for token in tokens:
                if token["kind"] == "motif":
                    output_tokens.append(token["value"])
                elif token["type"] == "open":
                    output_tokens.append(str(token["min"] + next(open_values)))
                else:
                    output_tokens.append(str(token["value"]))
            concrete_contigs.append("/".join(output_tokens))

    return concrete_contigs
