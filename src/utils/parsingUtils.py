import argparse

def parse_shape_savename_centrality(
    description="",
    centrality_choices=None
):
    """
    Parses:
      1) shape, as WIDTHxHEIGHT
      2) savename
      3) optional centrality (always present, default None,
         restricted to centrality_choices if provided)
    Returns: (x, y, savename, centrality)
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "shape",
        help="grid shape as WIDTHxHEIGHT, e.g. 300x300"
    )
    parser.add_argument(
        "savename",
        help="base name for your output (e.g. 'hello')"
    )
    # always add the positional 'centrality'
    if centrality_choices:
        parser.add_argument(
            "centrality",
            nargs="?",
            choices=centrality_choices,
            default=None,
            help="optional centrality measure; one of: "
                 + ", ".join(centrality_choices)
        )
    else:
        parser.add_argument(
            "centrality",
            nargs="?",
            default=None,
            help="optional third argument (no choices enforced)"
        )

    args = parser.parse_args()

    # parse shape → x,y
    try:
        w, h = args.shape.lower().split("x")
        x, y = int(w), int(h)
    except ValueError:
        parser.error("`shape` must be WIDTHxHEIGHT, e.g. 300x300")

    return x, y, args.savename, args.centrality
